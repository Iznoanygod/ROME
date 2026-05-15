"""Real GRPO smoke test.

Runs a single GRPO step on a tiny HuggingFace model. Skipped unless
torch/transformers/trl/datasets are installed. Marked `slow` so it stays
out of the default `pytest` invocation; opt in with `pytest -m slow`.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


def _import_or_skip():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    pytest.importorskip("peft")


def test_grpo_one_step_smoke(tmp_path):
    _import_or_skip()

    from datasets import Dataset

    from rome.config import ModelConfig
    from rome.train import GRPO

    # Trivial dataset of prompts. GRPO will generate completions itself.
    dataset = Dataset.from_list(
        [{"prompt": "The capital of France is"} for _ in range(2)]
    )

    def constant_reward(prompts, completions, **kwargs):
        return [1.0] * len(completions)

    from trl import GRPOConfig

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path / "grpo-out"),
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=8,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPO(
        gpus=1,
        dataset=dataset,
        reward_funcs=[constant_reward],
        rollout_func=None,
        grpo_config=grpo_config,
    )

    model_config = ModelConfig(
        base_model_name=os.environ.get(
            "ROME_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
        ),
    )

    # use_default_rollout=False so we skip our workflow-ddict-backed rollout
    # and let trl's built-in rollout run. workflow_ddict is unused in that
    # path but the signature requires something; an empty dict is fine.
    trainer.train(
        model_config=model_config,
        workflow_ddict={},
        use_default_rollout=False,
    )
