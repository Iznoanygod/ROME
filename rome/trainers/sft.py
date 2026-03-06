"""
rome/trainers/sft.py

SFTPolicy — Supervised Fine-Tuning training policy.

Used as the warm-start phase before RL, or standalone for imitation
learning from high-reward trajectories in the MemoryBank (RAFT/ReST style).
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from rome.trainer import Trainer, TrainerResult


class SFTPolicy(Trainer):
    """Supervised Fine-Tuning policy.

    Parameters
    ----------
    algorithm_cfg : SFTConfig | dict
    packing : bool
        Enable sequence packing for better GPU utilisation.
    dataset_text_field : str
        Column containing the training text (default: ``"text"``).
    max_seq_length : int
    checkpoint_dir : str
    seed : int
    use_unsloth : bool
    extra : dict
    """

    policy_name = "SFTPolicy"

    def __init__(
        self,
        algorithm_cfg: Any = None,
        packing: bool = False,
        dataset_text_field: str = "text",
        max_seq_length: int = 2048,
        checkpoint_dir: str = "./rome_checkpoints",
        seed: int = 42,
        use_unsloth: bool = True,
        extra: Optional[dict] = None,
    ) -> None:
        super().__init__(algorithm_cfg, checkpoint_dir, seed, use_unsloth, extra)
        self.packing = packing
        self.dataset_text_field = dataset_text_field
        self.max_seq_length = max_seq_length

    def train(
        self,
        model_path: str,
        tokenizer_path: str,
        dataset,
        reward_fns: Optional[List[Callable]] = None,
        iteration: int = 0,
    ) -> TrainerResult:
        if reward_fns:
            self.log("WARNING: reward_fns are ignored by SFTPolicy.")

        load_from = self.resolve_model_path(model_path, iteration)
        ckpt_path = self.checkpoint_path_for(iteration)
        os.makedirs(ckpt_path, exist_ok=True)
        self.log(f"iter={iteration} | loading from: {load_from}")

        model, tokenizer = self.load_model_and_tokenizer(load_from, self.max_seq_length)
        train_ds = self.load_dataset(dataset)
        train_ds = self._ensure_text_column(train_ds)
        sft_cfg = self._resolve_algorithm_config(ckpt_path)
        TrainerClass = self._resolve_trainer_class()

        self.log(
            f"Starting SFT | class={TrainerClass.__name__} | "
            f"steps={sft_cfg.max_steps} | packing={self.packing}"
        )

        trainer = TrainerClass(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            dataset_text_field=self.dataset_text_field,
            packing=self.packing,
            max_seq_length=self.max_seq_length,
        )
        trainer.train()

        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        self.log(f"Checkpoint saved → {ckpt_path}")

        _, loss, metrics = self.extract_log_metrics(trainer.state)
        self.log(f"loss={loss:.4f}")

        return TrainerResult(
            mean_reward=0.0,
            loss=loss,
            checkpoint_path=ckpt_path,
            metrics=metrics,
            iteration=iteration,
        )

    def _resolve_algorithm_config(self, output_dir: str):
        from trl import SFTConfig  # type: ignore
        if isinstance(self.algorithm_cfg, SFTConfig):
            return SFTConfig(**{**self.algorithm_cfg.to_dict(),
                                "output_dir": output_dir, "seed": self.seed})
        if isinstance(self.algorithm_cfg, dict):
            return SFTConfig(output_dir=output_dir, seed=self.seed,
                             **self.algorithm_cfg)
        return SFTConfig(
            output_dir=output_dir, seed=self.seed,
            max_steps=500, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4,
            warmup_ratio=0.03, lr_scheduler_type="cosine",
            logging_steps=20, bf16=True,
            max_seq_length=self.max_seq_length,
        )

    def _resolve_trainer_class(self):
        if self.use_unsloth:
            try:
                from unsloth import UnslothSFTTrainer  # type: ignore
                return UnslothSFTTrainer
            except ImportError:
                pass
        from trl import SFTTrainer  # type: ignore
        return SFTTrainer

    def _ensure_text_column(self, dataset):
        if self.dataset_text_field in dataset.column_names:
            return dataset
        if "prompt" in dataset.column_names and "completion" in dataset.column_names:
            self.log("Building 'text' column from 'prompt' + 'completion'.")
            dataset = dataset.map(
                lambda ex: {"text": ex["prompt"] + ex["completion"]}
            )
            self.dataset_text_field = "text"
        return dataset