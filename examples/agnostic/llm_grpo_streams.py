"""LLM self-improvement with all three ROME managers.

The counterpart to ``impress_r.py``. That example uses only the data and
training halves, because IMPRESS owns its own inference. This one uses the
stream manager too: generation and scoring run as persistent asynchronous
tasks, and they hot-swap onto each new LoRA adapter the trainer publishes —
without the workflow orchestrating the swap.

    inference stream  ──generations──▶  reward stream  ──scores──▶  Data Manager
             ▲                                                            │
             └──────────── checkpoint ◀── Training Manager ◀──────────────┘

Run under the Dragon runtime::

    dragon examples/agnostic/llm_grpo_streams.py
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import rome
from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig, load_model

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPTS = [
    "What is 17 * 23?",
    "A train leaves at 3pm going 60mph. How far in 2.5 hours?",
    "If x + 7 = 19, what is x?",
]


# ---------------------------------------------------------------------------
# The workflow's own inference and reward code. ROME just runs it in a loop.
# ---------------------------------------------------------------------------

model_config = ModelConfig(
    base_model_name=BASE_MODEL,
    lora_name="./adapters/math",
    required_gpus=1,
)


def load_generator(checkpoint_path, ctx):
    """Called at startup and again on every published checkpoint.

    Reloading is a LoRA adapter read, not a full model load, which is why a
    stream can swap weights between batches without stalling the campaign.
    """
    config = ModelConfig(
        base_model_name=BASE_MODEL,
        lora_name=checkpoint_path,
        required_gpus=1,
    )
    model, tokenizer = load_model(config)
    return {"model": model, "tokenizer": tokenizer}


def generate(prompts, ctx):
    """One batch of completions. Plain transformers — nothing ROME-specific."""
    import torch

    model, tokenizer = ctx.model["model"], ctx.model["tokenizer"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [
        {"prompt": prompt, "completion": completion}
        for prompt, completion in zip(prompts, completions)
    ]


def score(generations):
    """Reward function. Runs in its own stream, so it can be as slow as it likes.

    The dicts it returns become corpus records — the manager wires a reward
    stream's outputs into the data manager automatically.
    """
    scored = []
    for item in generations:
        completion = item["completion"]
        reward = 0.0
        if any(ch.isdigit() for ch in completion):
            reward += 1.0
        if len(completion) < 600:
            reward += 0.5
        scored.append({**item, "score": reward})
    return scored


# ---------------------------------------------------------------------------
# ROME adoption
# ---------------------------------------------------------------------------

async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)

    manager = rome.Manager(
        flow,
        data_config=rome.DataConfig(min_samples=32, sampling="top_k", shard_size=128),
        trainer_config=rome.TrainerConfig(
            trainer=GRPOTrainer(
                GRPOConfig(
                    model_config=model_config,
                    reward_funcs=[],   # scoring happens in the reward stream
                    num_generations=4,
                ),
            ),
            checkpoint_dir="./rome_checkpoints",
            poll_interval=5.0,
        ),
        stream_configs=[
            rome.StreamConfig(
                name="generate",
                kind=rome.StreamKind.INFERENCE,
                model_path=model_config.lora_name,
                load_func=load_generator,
                process_func=generate,
                num_streams=2,
                num_gpus=1,
                batch_size=4,
            ),
            rome.StreamConfig(
                name="score",
                kind=rome.StreamKind.REWARD,
                process_func=score,
                num_streams=2,
                num_gpus=0,
                batch_size=8,
            ),
        ],
    )
    await manager.start()

    try:
        for round_index in range(20):
            # Feed the inference stream, then hand its completions to the
            # reward stream. Scored results reach the corpus on their own.
            manager.stream.submit_batch(PROMPTS, stream="generate")

            for output in manager.stream.get_outputs(stream="generate"):
                manager.stream.submit(output["result"], stream="score")

            print(
                f"round {round_index}: corpus {manager.data.total_count} | "
                f"model v{manager.model_version} | "
                f"{manager.get_training_status().name} | "
                f"streams {[s.name for s in manager.get_stream_status()]}"
            )
            await asyncio.sleep(5.0)
    finally:
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
