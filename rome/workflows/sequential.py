"""
rome/workflows/sequential.py

SequentialWorkflow — single iterative RL self-improvement loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from radical.asyncflow import WorkflowEngine

from rome.config import ROMEConfig
from rome.workflow import BaseWorkflow

if TYPE_CHECKING:
    from rome.trainer import Trainer


class SequentialWorkflow(BaseWorkflow):
    """Single iterative RL workflow backed by ROSE's SequentialReinforcementLearner.

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

        await rome.launch(max_iter=10)
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
        super().__init__(asyncflow, policy=policy, config=config,
                         dataset=dataset, model=model)
        self._learner = None

    def _build_learner(self):
        from rose.rl.reinforcement_learner import SequentialReinforcementLearner

        learner = SequentialReinforcementLearner(self._asyncflow)
        policy  = self._config.policy

        # ── Simulation → environment_task ─────────────────────────────
        if self._simulation_fn is not None:
            simulation_fn = self._simulation_fn

            @learner.environment_task
            async def environment(*args):
                return await simulation_fn(*args)

        # ── Training → update_task ─────────────────────────────────────
        cfg = self._config

        @learner.update_task
        async def update(*args):
            iteration = int(args[0]) if args else 0
            result = policy.train(
                model_path=cfg.model,
                tokenizer_path=cfg.model,
                dataset=cfg.dataset,
                reward_fns=None,
                iteration=iteration,
            )
            print(result.mean_reward, flush=True)
            return result

        # ── Reward → stop criterion or utility task ────────────────────
        if self._reward_fn is not None:
            reward_fn = self._reward_fn
            threshold = self._config.reward_threshold

            if threshold is not None:
                from rose.metrics import GREATER_THAN_THRESHOLD

                @learner.as_stop_criterion(
                    metric_name="ROME_REWARD",
                    threshold=threshold,
                    operator=GREATER_THAN_THRESHOLD,
                )
                async def check_reward(*args):
                    return await reward_fn(*args)
            else:
                @learner.utility_task
                async def evaluate(*args):
                    return await reward_fn(*args)

        return learner

    async def launch(self, max_iter: Optional[int] = None) -> None:
        """Start the sequential RL loop.

        Parameters
        ----------
        max_iter : int, optional
            Override ``config.max_iterations``. 0 = run until reward
            threshold is met (requires ``reward_threshold`` to be set).
        """
        self._config.validate()
        iterations = (
            max_iter if max_iter is not None else self._config.max_iterations
        )
        self._learner = self._build_learner()

        print(
            f"[SequentialWorkflow] Launching | "
            f"policy={self._config.policy.__class__.__name__} | "
            f"dataset={self._config.dataset} | "
            f"max_iter={iterations or '∞'}",
            flush=True,
        )

        async for state in self._learner.start(max_iter=iterations):
            self._record_iteration(
                state.iteration,
                getattr(state, "metric_value", None),
            )