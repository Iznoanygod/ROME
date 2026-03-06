"""
rome/config.py

ROMEConfig — configuration for a ROME workflow run.

The ``policy`` field holds an instantiated Trainer policy object directly.
Users construct the policy themselves and pass it in:

    from rome.trainers import GRPOPolicy
    from trl import GRPOConfig

    policy = GRPOPolicy(
        algorithm_cfg=GRPOConfig(learning_rate=5e-5, max_steps=100),
    )
    config = ROMEConfig(policy=policy, dataset="...", model="...")

    # Or via attribute assignment on the workflow:
    rome.policy = GRPOPolicy(algorithm_cfg=GRPOConfig(...))
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rome.trainer import Trainer


@dataclasses.dataclass
class ROMEConfig:
    """Top-level configuration object for a ROME workflow.

    Attributes
    ----------
    policy : Trainer, optional
        An instantiated training policy (e.g. ``GRPOPolicy(...)``).
        Must be set before calling ``launch()``.
    dataset : str
        HuggingFace dataset name or ``"local:<path>"`` to training prompts.
    model : str
        Base model name/path (HuggingFace hub ID or local checkpoint path).
    max_iterations : int
        Maximum number of outer RL loop iterations.
        0 means run until the reward stop criterion is met.
    n_samples_per_prompt : int
        Number of completions generated per prompt (pass@k width).
    reward_threshold : float, optional
        If set, the workflow stops early when mean reward exceeds this value.
    checkpoint_dir : str
        Root directory for saving model checkpoints and evaluation artefacts.
    seed : int
        Global random seed for reproducibility.
    extra : dict
        Catch-all for domain-specific knobs (e.g. Rosetta scoring params).
    """

    policy: Optional["Trainer"] = None
    dataset: str = ""
    model: str = ""
    max_iterations: int = 0
    n_samples_per_prompt: int = 4
    reward_threshold: Optional[float] = None
    checkpoint_dir: str = "./rome_checkpoints"
    seed: int = 42
    extra: dict = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError for obviously invalid configurations."""
        if self.policy is None:
            raise ValueError(
                "ROMEConfig.policy must be set before launching. "
                "Example: rome.policy = GRPOPolicy(algorithm_cfg=GRPOConfig(...))"
            )
        if not self.dataset:
            raise ValueError("ROMEConfig.dataset must be set before launching.")
        if self.n_samples_per_prompt < 1:
            raise ValueError("n_samples_per_prompt must be >= 1.")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0.")