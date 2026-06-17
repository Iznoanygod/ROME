"""Basic StreamFlow demo — continuous-generation RL pipeline.

Mirrors the shape of ``protein_generation/proteinstream.py`` minus the
domain-specific folding/scoring tools: a pool of prompts is mirrored to
N generators that stream completions, scorers continuously consume those
completions to produce rewards, and the GRPO trainer pulls rolled-out
completions through its rollout function. The WeightSyncCallback publishes
a fresh LoRA adapter to ``checkpoint_dir`` after each training step;
generators reload between batches via ``maybe_reload_weights``.

Prerequisites
-------------
- Python 3.11 (required by Dragon)
- pip install dragonhpc rhapsody-py radical-asyncflow
- pip install transformers trl peft datasets

Launch
------
    dragon -s run.py

Known gap
---------
``GRPO.train()`` takes a positional ``dataset`` argument that
``StreamFlow.launch()`` does not yet pass through. Override the rollout
or patch the call site before relying on this for actual training.
"""
import asyncio
import logging

from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger
from rhapsody.backends import DragonExecutionBackendV3

from transformers import GenerationConfig

from rome.config import ModelConfig
from rome.flows.streamflow import StreamFlow, StreamFlowConfig
from rome.train import GRPO
from rome.workflow import Workflow


@Workflow.reward_task
def length_reward(completion):
    """Toy reward: length of the completion id list.

    The streaming scorer task hands each per-(prompt, idx) completion
    dict to this function; whatever it returns lands in
    ``reward_length_reward_outputs`` keyed by the same (prompt, idx).
    """
    completion_ids = completion.get("completion_ids", [])
    try:
        return float(len(completion_ids))
    except TypeError:
        return 0.0


async def evaluate(model_config):
    """Stop-criterion probe. Returning a static value lets the flow
    run for the configured number of iterations regardless of metric.
    """
    return 0.0


async def main():
    init_default_logger(logging.INFO)

    backend = await DragonExecutionBackendV3()
    asyncflow = await WorkflowEngine.create(backend=backend)

    model_config = ModelConfig(
        base_model_name="sshleifer/tiny-gpt2",
        lora_name="streamflow-demo-lora",
        generation_config=GenerationConfig(
            max_new_tokens=16,
            do_sample=True,
            top_k=20,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
        ),
    )

    trainer = GRPO(gpus=1, reward_funcs=[length_reward])

    stream_flow = StreamFlow(
        model_config=model_config,
        trainer=trainer,
        evaluate_func=evaluate,
        asyncflow=asyncflow,
        flow_config=StreamFlowConfig(
            iterations=2,
            num_generators=2,
            num_scorers=1,
            batch_size=2,
            prompts=["hello world", "the quick brown fox"],
            max_buffer_per_prompt=8,
            checkpoint_dir="./streamflow_demo_ckpts",
            checkpoint_interval=1,
        ),
    )

    await stream_flow.launch()
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
