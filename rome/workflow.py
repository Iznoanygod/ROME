"""
rome/workflow.py

BaseWorkflow — abstract base class for all ROME workflow types.

Subclasses:
    SequentialWorkflow  — rome/workflows/sequential.py
    ParallelWorkflow    — rome/workflows/parallel.py

Usage
-----
::

    from rome.trainers import GRPOPolicy
    from rome.workflows import SequentialWorkflow
    from trl import GRPOConfig

    policy = GRPOPolicy(algorithm_cfg=GRPOConfig(learning_rate=5e-5, max_steps=100))

    rome = SequentialWorkflow(asyncflow, policy=policy)
    rome.dataset = "qwedsacf/competition_math"
    rome.model   = "Qwen/Qwen2.5-1.5B-Instruct"

    @rome.simulation
    async def simulation(*args):
        return 'python3 generate_completions.py'

    @rome.reward
    async def reward(*args):
        return compute_score()

    await rome.launch()
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Optional

from radical.asyncflow import WorkflowEngine

from rome.config import ROMEConfig

if TYPE_CHECKING:
    from rome.trainer import Trainer


class BaseWorkflow(abc.ABC):
    """Abstract base class for all ROME workflow types.

    Parameters
    ----------
    asyncflow : WorkflowEngine
        Shared ``radical.asyncflow`` engine created by the user.
    policy : Trainer, optional
        An instantiated training policy (e.g. ``GRPOPolicy(...)``).
        Can also be set after construction via ``rome.policy = ...``.
    config : ROMEConfig, optional
        Pre-built configuration object. Any fields set here are
        overridden by explicit keyword arguments.
    dataset : str, optional
        HuggingFace dataset name or ``"local:<path>"``.
    model : str, optional
        Base model hub ID or local path.
    """

    def __init__(
        self,
        asyncflow: WorkflowEngine,
        policy: Optional["Trainer"] = None,
        config: Optional[ROMEConfig] = None,
        *,
        dataset: str = "",
        model: str = "",
    ) -> None:
        self._asyncflow = asyncflow
        self._config: ROMEConfig = config or ROMEConfig()

        # Explicit constructor kwargs override whatever is in config
        if policy is not None:
            self._config.policy = policy
        if dataset:
            self._config.dataset = dataset
        if model:
            self._config.model = model

        self._simulation_fn: Optional[Callable] = None
        self._reward_fn: Optional[Callable] = None
        self._iteration_history: list[dict[str, Any]] = []

    # ── Decorator API ────────────────────────────────────────────────────

    def simulation(self, fn: Callable) -> Callable:
        """Register the simulation / completion-generation function.

        The decorated async function should return a shell command string
        (dispatched to HPC via ROSE) or an artefact path.

        Example
        -------
        ::

            @rome.simulation
            async def simulation(*args):
                return 'python3 generate_completions.py'
        """
        self._simulation_fn = fn
        return fn

    def reward(self, fn: Callable) -> Callable:
        """Register the reward / scoring function.

        Must return a ``float``. ROME forwards this to ROSE as the
        iteration metric and stop-criterion value.

        Example
        -------
        ::

            @rome.reward
            async def reward(*args):
                return compute_score()
        """
        self._reward_fn = fn
        return fn

    # ── Config shim ───────────────────────────────────────────────────────

    @property
    def config(self) -> ROMEConfig:
        """The underlying :class:`ROMEConfig` object."""
        return self._config

    @property
    def policy(self) -> Optional["Trainer"]:
        """The instantiated training policy."""
        return self._config.policy

    @policy.setter
    def policy(self, value: "Trainer") -> None:
        self._config.policy = value

    @property
    def dataset(self) -> str:
        return self._config.dataset

    @dataset.setter
    def dataset(self, value: str) -> None:
        self._config.dataset = value

    @property
    def model(self) -> str:
        return self._config.model

    @model.setter
    def model(self, value: str) -> None:
        self._config.model = value

    # ── Abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def _build_learner(self):
        """Instantiate and wire the underlying ROSE learner(s).

        Called once inside ``launch()`` after config validation.
        """

    @abc.abstractmethod
    async def launch(self, **kwargs) -> Any:
        """Start the ROME RL loop."""

    # ── Shared internals ─────────────────────────────────────────────────

    def _record_iteration(self, iteration: int, metric_value: Any) -> None:
        record = {"iteration": iteration, "metric_value": metric_value}
        self._iteration_history.append(record)
        print(
            f"[{self.__class__.__name__}] "
            f"iter={iteration:4d} | reward={metric_value}",
            flush=True,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def history(self) -> list[dict[str, Any]]:
        """Per-iteration records: list of ``{iteration, metric_value}`` dicts."""
        return list(self._iteration_history)

    def summary(self) -> str:
        """Human-readable summary of the completed run."""
        if not self._iteration_history:
            return f"[{self.__class__.__name__}] No iterations completed yet."
        rewards = [
            h["metric_value"]
            for h in self._iteration_history
            if h["metric_value"] is not None
        ]
        best = max(rewards) if rewards else None
        n = len(self._iteration_history)
        return (
            f"[{self.__class__.__name__}] Completed {n} iteration(s) | "
            + (f"best reward={best:.4f}" if best is not None else "no rewards recorded")
        )

    def __repr__(self) -> str:
        policy_name = (
            self._config.policy.__class__.__name__
            if self._config.policy else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"policy={policy_name}, "
            f"dataset={self._config.dataset!r}, "
            f"iterations={len(self._iteration_history)})"
        )