"""
rome/trainers/grpo.py

GRPOPolicy — Group Relative Policy Optimization training policy.

GRPO is the primary online RL algorithm for ROME (used in DeepSeek-R1
and ROME's math reasoning experiments). It generates a group of
completions per prompt, scores them with reward functions, and uses
the relative group scores as the advantage estimate — no critic needed.

"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from rome.trainer import Trainer, TrainerResult


class GRPOPolicy(Trainer):
    """GRPO training policy.

    Parameters
    ----------
    algorithm_cfg : GRPOConfig | dict
        TRL ``GRPOConfig`` or plain dict of kwargs.
    inject_format_reward : bool
        Prepend ``rome.rewards.format_reward`` to user reward functions.
    use_vllm : bool
        Use vLLM for the rollout generation phase.
    checkpoint_dir : str
    seed : int
    use_unsloth : bool
    extra : dict
    """

    policy_name = "GRPOPolicy"

    def __init__(
        self,
        algorithm_cfg: Any = None,
        inject_format_reward: bool = True,
        use_vllm: bool = False,
        checkpoint_dir: str = "./rome_checkpoints",
        seed: int = 42,
        use_unsloth: bool = True,
        extra: Optional[dict] = None,
    ) -> None:
        super().__init__(algorithm_cfg, checkpoint_dir, seed, use_unsloth, extra)
        self.inject_format_reward = inject_format_reward
        self.use_vllm = use_vllm

    # ── train ─────────────────────────────────────────────────────────────

    def train(
        self,
        model_path: str,
        tokenizer_path: str,
        dataset,
        reward_fns: Optional[List[Callable]] = None,
        iteration: int = 0,
    ) -> TrainerResult:
        load_from = self.resolve_model_path(model_path, iteration)
        ckpt_path = self.checkpoint_path_for(iteration)
        os.makedirs(ckpt_path, exist_ok=True)
        self.log(f"iter={iteration} | loading from: {load_from}")

        model, tokenizer = self.load_model_and_tokenizer(load_from)
        train_ds = self.load_dataset(dataset)
        all_reward_fns = self._build_reward_fns(reward_fns)
        grpo_cfg = self._resolve_algorithm_config(ckpt_path)
        TrainerClass = self._resolve_trainer_class()

        self.log(
            f"Starting GRPO | class={TrainerClass.__name__} | "
            f"steps={grpo_cfg.max_steps} | "
            f"n_completions={getattr(grpo_cfg, 'num_generations', 8)}"
        )

        trainer = TrainerClass(
            model=model,
            reward_funcs=all_reward_fns,
            args=grpo_cfg,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )
        trainer.train()

        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        self.log(f"Checkpoint saved → {ckpt_path}")

        mean_reward, loss, metrics = self.extract_log_metrics(trainer.state)
        self.log(f"mean_reward={mean_reward:.4f} | loss={loss:.4f}")

        return TrainerResult(
            mean_reward=mean_reward,
            loss=loss,
            checkpoint_path=ckpt_path,
            metrics=metrics,
            iteration=iteration,
        )

    # ── Abstract implementations ──────────────────────────────────────────

    def _resolve_algorithm_config(self, output_dir: str):
        from trl import GRPOConfig  # type: ignore
        if isinstance(self.algorithm_cfg, GRPOConfig):
            return GRPOConfig(**{**self.algorithm_cfg.to_dict(),
                                 "output_dir": output_dir, "seed": self.seed})
        if isinstance(self.algorithm_cfg, dict):
            return GRPOConfig(output_dir=output_dir, seed=self.seed,
                              **self.algorithm_cfg)
        return GRPOConfig(
            output_dir=output_dir, seed=self.seed,
            max_steps=200, per_device_train_batch_size=4,
            num_generations=8, learning_rate=5e-6,
            gradient_accumulation_steps=2, logging_steps=10,
            bf16=True, remove_unused_columns=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_reward_fns(self, user_fns: Optional[List[Callable]]) -> List[Callable]:
        fns: List[Callable] = []
        if self.inject_format_reward:
            from rome.rewards import format_reward as _fmt

            def _trl_format(completions, **kw):
                return [_fmt(c) for c in completions]
            _trl_format.__name__ = "format_reward"
            fns.append(_trl_format)
        if user_fns:
            fns.extend(user_fns)
        if not fns:
            self.log("WARNING: no reward functions — using zero reward.")
            fns = [lambda completions, **kw: [0.0] * len(completions)]
        return fns

    def _resolve_trainer_class(self):
        if self.use_unsloth:
            try:
                from unsloth import UnslothGRPOTrainer  # type: ignore
                return UnslothGRPOTrainer
            except ImportError:
                pass
        from trl import GRPOTrainer  # type: ignore
        return GRPOTrainer