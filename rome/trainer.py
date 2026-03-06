"""
rome/trainer.py

Trainer — abstract base class for all ROME training policies.

This is the root of the ROME trainer hierarchy. All training
policies inherit from this class and must implement ``train()`` and
``_resolve_algorithm_config()``.

Policies in rome/trainers/:
    GRPOPolicy   — rome/trainers/grpo.py
    SFTPolicy    — rome/trainers/sft.py
    PPOPolicy    — rome/trainers/ppo.py
    DPOPolicy    — rome/trainers/dpo.py
    RRHFPolicy   — rome/trainers/rrhf.py

"""

from __future__ import annotations

import abc
import dataclasses
import os
from typing import Any, Callable, List, Optional


# ── Result container ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainerResult:
    """Standardised return value from one training iteration.

    Attributes
    ----------
    mean_reward : float
        Mean reward over the training batch. 0.0 for supervised methods.
    loss : float
        Final training loss.
    checkpoint_path : str
        Absolute path to the saved checkpoint directory.
    metrics : dict
        Additional algorithm-specific metrics (lr, grad_norm, kl_div, …).
    iteration : int
        The outer ROME loop iteration this result belongs to.
    """
    mean_reward: float = 0.0
    loss: float = 0.0
    checkpoint_path: str = ""
    metrics: dict = dataclasses.field(default_factory=dict)
    iteration: int = 0

    def __str__(self) -> str:
        return (
            f"TrainerResult(iter={self.iteration}, "
            f"reward={self.mean_reward:.4f}, loss={self.loss:.4f}, "
            f"ckpt={self.checkpoint_path})"
        )


# ── Abstract base ─────────────────────────────────────────────────────────────

class Trainer(abc.ABC):
    """Abstract base class for ROME training policies.

    Every concrete policy (GRPO, SFT, PPO, DPO, RRHF) inherits from
    this class and implements ``train()`` and ``_resolve_algorithm_config()``.
    Everything else — checkpoint management, model loading, dataset loading,
    logging, HPC serialisation — is provided here so subclasses stay focused
    on their algorithm logic.

    Parameters
    ----------
    algorithm_cfg : Any
        Algorithm-specific config object (e.g. ``GRPOConfig``, ``SFTConfig``).
        Accepts both the dataclass form and a plain dict (for HPC node
        reconstruction from serialised JSON).
    checkpoint_dir : str
        Root directory for checkpoints. Each iteration writes to
        ``{checkpoint_dir}/iter_{N:04d}/``.
    seed : int
        Random seed forwarded to the underlying TRL trainer.
    use_unsloth : bool
        When True (default), attempt to load model via Unsloth's
        ``FastLanguageModel`` for accelerated training. Falls back to
        standard HuggingFace if Unsloth is not installed.
    extra : dict
        Catch-all for algorithm-specific knobs not in ``algorithm_cfg``
        (e.g. ``use_vllm``, ``beta``, ``n_completions``).
    """

    #: Human-readable policy name used in logs. Override in subclasses.
    policy_name: str = "Trainer"

    def __init__(
        self,
        algorithm_cfg: Any = None,
        checkpoint_dir: str = "./rome_checkpoints",
        seed: int = 42,
        use_unsloth: bool = True,
        extra: Optional[dict] = None,
    ) -> None:
        self.algorithm_cfg = algorithm_cfg
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.use_unsloth = use_unsloth
        self.extra: dict = extra or {}

    # ── Abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def train(
        self,
        model_path: str,
        tokenizer_path: str,
        dataset,
        reward_fns: Optional[List[Callable]] = None,
        iteration: int = 0,
    ) -> TrainerResult:
        """Execute one training iteration.

        Parameters
        ----------
        model_path : str
            HF hub ID or local path for the base model (iteration 0), or
            path to the previous checkpoint (iteration N).
        tokenizer_path : str
            Path to the tokenizer. Usually identical to ``model_path``.
        dataset :
            HuggingFace ``Dataset``, ``DatasetDict``, or a string path.
            Online policies (GRPO, PPO) need a ``prompt`` column.
            Offline policies (SFT, DPO, RRHF) need policy-specific columns.
        reward_fns : list[Callable], optional
            Reward functions with TRL signature:
            ``fn(completions: list[str], **kwargs) -> list[float]``.
            Required for online RL policies (GRPO, PPO).
            Pass ``None`` for offline policies (SFT, DPO, RRHF).
        iteration : int
            Current outer ROME loop iteration (used for checkpoint naming
            and checkpoint-chain resolution).

        Returns
        -------
        TrainerResult
        """

    @abc.abstractmethod
    def _resolve_algorithm_config(self, output_dir: str) -> Any:
        """Build or adapt the algorithm config for this iteration.

        Called inside ``train()`` with the iteration's checkpoint directory
        as ``output_dir``. Returns a TRL *Config object ready to pass to
        the underlying trainer.

        Accepts both the dataclass form and a plain dict (for HPC nodes
        that deserialised from JSON).
        """

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def checkpoint_path_for(self, iteration: int) -> str:
        """Return ``{checkpoint_dir}/iter_{iteration:04d}``."""
        return os.path.join(self.checkpoint_dir, f"iter_{iteration:04d}")

    def latest_checkpoint(self, before_iteration: int) -> Optional[str]:
        """Walk backwards to find the most recent existing checkpoint.

        Returns ``None`` if no prior checkpoint exists (e.g. first run).
        Enables seamless HPC restart recovery.
        """
        for i in range(before_iteration - 1, -1, -1):
            path = self.checkpoint_path_for(i)
            if os.path.isdir(path):
                return path
        return None

    def resolve_model_path(self, base_model: str, iteration: int) -> str:
        """Return the correct model path to load for this iteration.

        - Iteration 0 with no prior checkpoint → ``base_model``
        - Any iteration with a prior checkpoint → that checkpoint path
        """
        prev = self.latest_checkpoint(iteration)
        return prev if prev is not None else base_model

    # ── Model + tokenizer loading ─────────────────────────────────────────

    def load_model_and_tokenizer(self, path: str, max_seq_length: int = 2048):
        """Load model and tokenizer, preferring Unsloth if available.

        Returns ``(model, tokenizer)``.
        """
        if self.use_unsloth:
            try:
                from unsloth import FastLanguageModel  # type: ignore
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=16,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=16,
                    lora_dropout=0.0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                )
                self.log(f"Loaded via Unsloth FastLanguageModel + LoRA: {path}")
                return model, tokenizer
            except ImportError:
                self.log("Unsloth not available — falling back to HuggingFace.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.log(f"Loaded via HuggingFace: {path}")
        return model, tokenizer

    # ── Dataset loading ───────────────────────────────────────────────────

    @staticmethod
    def load_dataset(dataset):
        """Accept HuggingFace Dataset, DatasetDict, HF hub name, or local path.

        Local paths must be prefixed with ``"local:"``
        (e.g. ``"local:data/prompts.jsonl"``).
        """
        if isinstance(dataset, str):
            from datasets import load_dataset as _ld  # type: ignore
            if dataset.startswith("local:"):
                ds = _ld("json", data_files=dataset[len("local:"):])
            else:
                ds = _ld(dataset)
            # Unwrap DatasetDict → train split
            if hasattr(ds, "keys"):
                return ds.get("train", ds[list(ds.keys())[0]])
            return ds
        try:
            from datasets import DatasetDict  # type: ignore
            if isinstance(dataset, DatasetDict):
                return dataset.get("train", dataset[list(dataset.keys())[0]])
        except ImportError:
            pass
        return dataset

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        print(f"[{self.policy_name}] {msg}", flush=True)

    # ── Metrics extraction helper ─────────────────────────────────────────

    @staticmethod
    def extract_log_metrics(trainer_state) -> tuple[float, float, dict]:
        """Pull mean_reward, loss, and extras from a TRL trainer's log history.

        Returns ``(mean_reward, loss, extra_metrics)``.
        """
        mean_reward = 0.0
        loss = 0.0
        extra: dict = {}
        if not (hasattr(trainer_state, "log_history") and trainer_state.log_history):
            return mean_reward, loss, extra
        for entry in reversed(trainer_state.log_history):
            if "reward" in entry and mean_reward == 0.0:
                mean_reward = float(entry["reward"])
            if "loss" in entry and loss == 0.0:
                loss = float(entry["loss"])
            if mean_reward and loss:
                break
        if trainer_state.log_history:
            last = trainer_state.log_history[-1]
            extra = {k: v for k, v in last.items() if k not in ("reward", "loss")}
        return mean_reward, loss, extra