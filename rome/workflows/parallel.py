"""
rome/workflows/parallel.py

ParallelWorkflow — N independent RL workflows running concurrently.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING, Any, List, Optional

from radical.asyncflow import WorkflowEngine

from rome.config import ROMEConfig
from rome.workflow import BaseWorkflow

if TYPE_CHECKING:
    from rome.trainer import Trainer


class WorkflowResult:
    """Result from one sub-workflow in a ParallelWorkflow run.

    Attributes
    ----------
    workflow_id : int
    config : ROMEConfig
    history : list[dict]
        Per-iteration records: ``{iteration, metric_value}``.
    error : Exception | None
    """

    def __init__(
        self,
        workflow_id: int,
        config: ROMEConfig,
        history: list,
        error: Optional[Exception] = None,
    ) -> None:
        self.workflow_id = workflow_id
        self.config      = config
        self.history     = history
        self.error       = error

    @property
    def final_metric(self) -> Optional[float]:
        return self.history[-1].get("metric_value") if self.history else None

    @property
    def best_metric(self) -> Optional[float]:
        rewards = [h["metric_value"] for h in self.history
                   if h.get("metric_value") is not None]
        return max(rewards) if rewards else None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        policy_name = (
            self.config.policy.__class__.__name__
            if self.config.policy else "None"
        )
        status = (
            f"best={self.best_metric:.4f}"
            if self.succeeded and self.best_metric is not None
            else f"ERROR: {self.error}"
        )
        return (
            f"WorkflowResult(id={self.workflow_id}, "
            f"policy={policy_name}, {status})"
        )


class ParallelWorkflow(BaseWorkflow):
    """N concurrent independent RL workflows.

    Usage — homogeneous (same policy, different seeds):
    ---------------------------------------------------
    ::

        policy = GRPOPolicy(algorithm_cfg=GRPOConfig(...))
        rome   = ParallelWorkflow(asyncflow, policy=policy, n_workflows=4)
        rome.dataset = "qwedsacf/competition_math"
        rome.model   = "Qwen/Qwen2.5-1.5B-Instruct"

    Usage — heterogeneous (different policy per workflow):
    ------------------------------------------------------
    ::

        configs = [
            ROMEConfig(policy=GRPOPolicy(...), model="Qwen/1.5B", dataset="...", seed=0),
            ROMEConfig(policy=SFTPolicy(...),  model="Qwen/3B",   dataset="...", seed=1),
        ]
        rome = ParallelWorkflow(asyncflow, workflow_configs=configs)

    Parameters
    ----------
    asyncflow : WorkflowEngine
    policy : Trainer, optional
        Default policy for sub-workflows without a per-workflow config.
    n_workflows : int
        Number of concurrent sub-workflows (ignored when workflow_configs given).
    config : ROMEConfig, optional
    workflow_configs : list[ROMEConfig | None], optional
        Per-workflow config overrides. None entries fall back to default config.
    dataset : str, optional
    model : str, optional
    """

    def __init__(
        self,
        asyncflow: WorkflowEngine,
        policy: Optional["Trainer"] = None,
        n_workflows: int = 2,
        config: Optional[ROMEConfig] = None,
        workflow_configs: Optional[List[Optional[ROMEConfig]]] = None,
        *,
        dataset: str = "",
        model: str = "",
    ) -> None:
        super().__init__(asyncflow, policy=policy, config=config,
                         dataset=dataset, model=model)

        if workflow_configs is not None:
            self._workflow_configs: List[Optional[ROMEConfig]] = workflow_configs
            self._n_workflows = len(workflow_configs)
        else:
            self._n_workflows = n_workflows
            self._workflow_configs = [None] * n_workflows

        self._results: List[WorkflowResult] = []

    def _resolve_config(self, workflow_id: int) -> ROMEConfig:
        override = self._workflow_configs[workflow_id]
        return override if override is not None else self._config

    def set_workflow_config(self, workflow_id: int, cfg: ROMEConfig) -> None:
        """Assign a per-workflow config override after construction."""
        if workflow_id >= self._n_workflows:
            raise IndexError(
                f"workflow_id={workflow_id} out of range "
                f"(n_workflows={self._n_workflows})"
            )
        self._workflow_configs[workflow_id] = cfg

    def _build_sub_learner(self, workflow_id: int):
        from rose.rl.reinforcement_learner import SequentialReinforcementLearner

        cfg = dataclasses.replace(
            self._resolve_config(workflow_id),
            checkpoint_dir=(
                f"{self._resolve_config(workflow_id).checkpoint_dir}"
                f"/workflow_{workflow_id:02d}"
            ),
        )
        policy = cfg.policy

        learner = SequentialReinforcementLearner(self._asyncflow)
        learner.learner_id = workflow_id

        # Simulation
        if self._simulation_fn is not None:
            simulation_fn = self._simulation_fn

            @learner.environment_task
            async def environment(*args):
                return await simulation_fn(*args)

        # Training
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

        # Reward
        if self._reward_fn is not None:
            reward_fn = self._reward_fn
            threshold = cfg.reward_threshold

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

    def _build_learner(self):
        raise NotImplementedError(
            "ParallelWorkflow builds per-workflow learners via _build_sub_learner()."
        )

    async def launch(
        self,
        max_iter: Optional[int] = None,
        fail_fast: bool = False,
    ) -> List[WorkflowResult]:
        """Start all N sub-workflows concurrently.

        Parameters
        ----------
        max_iter : int, optional
            Override max_iterations for all sub-workflows.
        fail_fast : bool
            Cancel remaining workflows on first failure.

        Returns
        -------
        list[WorkflowResult]
        """
        for i in range(self._n_workflows):
            self._resolve_config(i).validate()

        iterations = (
            max_iter if max_iter is not None else self._config.max_iterations
        )
        sub_learners = [self._build_sub_learner(i) for i in range(self._n_workflows)]

        print(
            f"[ParallelWorkflow] Launching {self._n_workflows} workflows | "
            f"max_iter={iterations or '∞'}",
            flush=True,
        )

        async def _run(workflow_id: int) -> WorkflowResult:
            learner = sub_learners[workflow_id]
            cfg     = self._resolve_config(workflow_id)
            history: list = []
            error: Optional[Exception] = None

            policy_name = cfg.policy.__class__.__name__ if cfg.policy else "None"
            print(
                f"[ParallelWorkflow] wf={workflow_id} starting | "
                f"policy={policy_name}",
                flush=True,
            )

            try:
                async for state in learner.start(max_iter=iterations):
                    record = {
                        "iteration":    state.iteration,
                        "metric_value": getattr(state, "metric_value", None),
                    }
                    history.append(record)
                    self._iteration_history.append(
                        {"workflow_id": workflow_id, **record}
                    )
                    print(
                        f"[ParallelWorkflow] wf={workflow_id} | "
                        f"iter={state.iteration:4d} | "
                        f"reward={record['metric_value']}",
                        flush=True,
                    )
            except Exception as exc:
                error = exc
                print(f"[ParallelWorkflow] wf={workflow_id} FAILED: {exc}", flush=True)
                if fail_fast:
                    raise

            return WorkflowResult(
                workflow_id=workflow_id, config=cfg,
                history=history, error=error,
            )

        raw = await asyncio.gather(
            *[_run(i) for i in range(self._n_workflows)],
            return_exceptions=not fail_fast,
        )

        self._results = [
            r if isinstance(r, WorkflowResult)
            else WorkflowResult(
                workflow_id=i,
                config=self._resolve_config(i),
                history=[],
                error=r,
            )
            for i, r in enumerate(raw)
        ]

        succeeded = sum(1 for r in self._results if r.succeeded)
        print(
            f"[ParallelWorkflow] Done — {succeeded}/{self._n_workflows} succeeded.",
            flush=True,
        )
        return self._results

    def best(self) -> Optional[WorkflowResult]:
        """Return the WorkflowResult with the highest best_metric."""
        candidates = [r for r in self._results
                      if r.succeeded and r.best_metric is not None]
        return max(candidates, key=lambda r: r.best_metric) if candidates else None

    def summary(self) -> str:
        if not self._results:
            return "[ParallelWorkflow] No results yet."
        lines = [f"[ParallelWorkflow] {self._n_workflows} workflows:"]
        for r in self._results:
            policy_name = (
                r.config.policy.__class__.__name__
                if r.config.policy else "None"
            )
            status = (
                f"best={r.best_metric:.4f}" if r.best_metric is not None
                else "no reward"
            )
            err = f" ERROR: {r.error}" if not r.succeeded else ""
            lines.append(f"  wf-{r.workflow_id}: {status} | policy={policy_name}{err}")
        best = self.best()
        if best:
            lines.append(
                f"  → Best: wf-{best.workflow_id} (reward={best.best_metric:.4f})"
            )
        return "\n".join(lines)

    @property
    def results(self) -> List[WorkflowResult]:
        return list(self._results)

    @property
    def n_workflows(self) -> int:
        return self._n_workflows