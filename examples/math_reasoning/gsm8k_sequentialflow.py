"""GSM8K math-reasoning fine-tuning via ROME's :class:`SequentialFlow`.

Equivalent to the standalone ``reward.py`` / ``update.py`` scripts in this
directory but routed through ROME's asyncflow-backed sequential RL loop.
The generator + trainer run as separate asyncflow tasks under a shared
:class:`radical.asyncflow.WorkflowEngine`, coordinated through a Dragon
DDict — the same shared-state substrate ROME uses in the protein flow.

Layout::

    +--------------------+       +------------+       +----------------+
    |  N × generator     |<----->| workflow   |<----->| GRPO trainer   |
    |  tasks (LLM.gen)   |       | DDict      |       | (trl.GRPOTrainer)
    +--------------------+       +------------+       +----------------+
                                    ^  ^  ^
                                    |  |  |
                          scheduler + gatherer coroutines
                            (rome.flows.SequentialFlow)

Reward functions are the same GSM8K format + numeric-answer checks used
in the standalone scripts. They're inlined here so the file is
self-contained; import from ``reward.py`` instead if you want to keep
one source of truth for both entry points.

Run on a GPU node — this is a heavyweight script that loads Llama-3.2-3B
+ a LoRA adapter and pulls the openai/gsm8k dataset. It's not part of
the fast test suite (``pytest -m fast`` skips it).
"""

import asyncio
import os
import re
from typing import Optional

# Optional Colab-style HF_TOKEN plumbing that mirrors the standalone scripts.
try:
    from google.colab import userdata  # type: ignore
    os.environ["HF_TOKEN"] = userdata.get("hf_token")
except Exception:
    os.environ.setdefault("HF_TOKEN", "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import GenerationConfig
from trl import GRPOConfig

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend  # type: ignore

from rome.config import ModelConfig
from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig
from rome.train.grpo import GRPO
from rome.utils import load_model


# ---------------------------------------------------------------------------
# GSM8K prompt template + reward functions (inlined from reward.py so this
# file is self-contained).
# ---------------------------------------------------------------------------

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
MATCH_NUMBERS = re.compile(
    SOLUTION_START + r".*?([\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)


def _content(completion):
    """Normalize a completion — trl passes either a chat-style list of
    ``{'role': ..., 'content': ...}`` dicts or a raw string, depending on
    the model config. Reduce to the raw response text.
    """
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return completion


def match_format_exactly(completions, **kwargs):
    scores = []
    for c in completions:
        scores.append(3.0 if MATCH_FORMAT.search(_content(c)) is not None else 0.0)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for c in completions:
        response = _content(c)
        score = 0.0
        score += 0.5 if response.count(REASONING_START) == 1 else -1.0
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = [_content(c) for c in completions]
    extracted = [
        m.group(1) if (m := MATCH_FORMAT.search(r)) else None for r in responses
    ]
    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        if guess == truth:
            scores.append(3.0)
        elif guess.strip() == truth.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(truth)
                if 0.9 <= ratio <= 1.1:
                    scores.append(1.0)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.5)
                else:
                    scores.append(-1.5)
            except Exception:
                scores.append(-1.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    responses = [_content(c) for c in completions]
    extracted = [
        m.group(1) if (m := MATCH_NUMBERS.search(r)) else None for r in responses
    ]
    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            true_answer = float(truth.strip())
            guessed = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guessed == true_answer else -0.5)
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Dataset preparation — apply the chat template + extract the numeric answer.
# ---------------------------------------------------------------------------

def _extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def load_train_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": _extract_hash_answer(x["answer"]),
        }
    )


def load_test_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "ground_truth": _extract_hash_answer(x["answer"]),
        }
    )


# ---------------------------------------------------------------------------
# Per-iteration evaluation — mirrors reward.py but returns a scalar so
# ROSE's stop criterion can compare it against reward_threshold.
# ---------------------------------------------------------------------------

async def evaluate_gsm8k(
    model_config: ModelConfig,
    num_questions: int = 16,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
) -> float:
    """Run a small held-out eval and return the mean total reward per
    question. Wrapper is async so it slots into SequentialFlow's stop
    criterion machinery.
    """
    model, tokenizer = load_model(model_config)
    test = load_test_dataset().shuffle(seed=42)
    questions = test.select(range(min(num_questions, len(test))))

    total_score = 0.0
    for question in questions:
        messages = [question["prompt"]] * batch_size
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:], skip_special_tokens=True
        )
        ground_truths = [question["ground_truth"]] * batch_size
        rewards = (
            match_format_exactly(responses)
            + match_format_approximately(responses)
            + check_answer(messages, responses, ground_truths)
            + check_numbers(messages, responses, ground_truths)
        )
        total_score += sum(rewards) / batch_size

    return total_score / len(questions)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    # ---- Model + generation config -------------------------------------
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    generation_cfg = GenerationConfig(
        max_new_tokens=2048 - 288,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        return_dict_in_generate=True,
        output_scores=True,
    )
    model_config = ModelConfig(
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        lora_name="32llamalora",
        lora_config=lora_cfg,
        generation_config=generation_cfg,
        dtype="bfloat16",
        required_gpus=1,
        max_seq_length=2048,
    )

    # ---- Trainer (GRPO) — dataset comes in through the constructor now,
    #      so SequentialFlow's train_model task can await
    #      trainer.train(model_config, workflow_ddict=...) without
    #      needing to plumb the dataset through the update task.
    train_dataset = load_train_dataset()
    grpo_config = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=288,
        max_completion_length=2048 - 288,
        max_steps=100,
        save_steps=50,
        max_grad_norm=1.0,
        run_name="rome-gsm8k-sequentialflow",
        output_dir="32llamalora",
        overwrite_output_dir=True,
    )
    trainer = GRPO(
        gpus=1,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        dataset=train_dataset,
        grpo_config=grpo_config,
    )

    # ---- AsyncFlow backend + workflow engine ---------------------------
    # ConcurrentExecutionBackend runs tasks in the local process; swap for
    # RadicalExecutionBackend when running on an HPC allocation.
    backend = ConcurrentExecutionBackend()
    asyncflow = await WorkflowEngine.create(backend=backend)

    # ---- Wire the flow -------------------------------------------------
    flow_config = SequentialFlowConfig(
        iterations=int(os.environ.get("ROME_ITERATIONS", 5)),
        reward_threshold=float(os.environ.get("ROME_REWARD_THRESHOLD", 12.0)),
        num_generators=int(os.environ.get("ROME_NUM_GENERATORS", 2)),
        num_scorers=int(os.environ.get("ROME_NUM_SCORERS", 2)),
        batch_size=int(os.environ.get("ROME_BATCH_SIZE", 4)),
    )
    flow = SequentialFlow(
        model_config=model_config,
        trainer=trainer,
        evaluate_func=evaluate_gsm8k,
        asyncflow=asyncflow,
        flow_config=flow_config,
    )

    await flow.launch()
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
