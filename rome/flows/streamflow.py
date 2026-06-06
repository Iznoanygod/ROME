"""StreamFlow scaffold.

StreamFlow streams generations into a pool of scorers. Per the design notes,
the principal batching knob is the *scorer* batch size — generations arrive
continuously, scorers accumulate them and run scoring in batches. This file
ships the configuration and the integration hooks (dynamic batch read,
throughput monitor) that an automatic batch-size controller will use; the
full streaming loop and the controller itself land in follow-ups.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from rose.metrics import GREATER_THAN_THRESHOLD

from rome.config import ModelConfig
from rome.trainer import Trainer
from rome.tuning import BatchTuningConfig, ThroughputMonitor, ThroughputSample
from rome.workflow import Workflow


class StreamFlowConfig:
    """Configuration for StreamFlow.

    Parameters
    ----------
    iterations : int
        Number of RL iterations to run. ``0`` runs until the reward threshold
        is met (requires ``reward_threshold`` to be set).
    reward_threshold : float, optional
        Stop criterion threshold.
    operator : str
        Comparison operator for the stop criterion.
    num_generators : int
        Number of generator tasks producing the generation stream.
    num_scorers : int
        Number of scorer tasks per reward function.
    scorer_batch_size : int
        Batch size used by scorer tasks when tuning is disabled, and the
        starting batch size when it is enabled (overridden by
        ``tuning.initial_batch_size`` when ``tuning.enabled``).
    tuning : BatchTuningConfig, optional
        Config for the automatic batch-size tuner. Defaults to a disabled
        tuner; flows still record throughput samples for offline analysis.
    """

    def __init__(
        self,
        iterations: int = 10,
        reward_threshold: Optional[float] = None,
        operator: str = GREATER_THAN_THRESHOLD,
        num_generators: int = 2,
        num_scorers: int = 2,
        scorer_batch_size: int = 4,
        tuning: Optional[BatchTuningConfig] = None,
    ):
        self.iterations = iterations
        self.reward_threshold = reward_threshold
        self.operator = operator
        self.num_generators = num_generators
        self.num_scorers = num_scorers
        self.scorer_batch_size = scorer_batch_size
        self.tuning = tuning if tuning is not None else BatchTuningConfig()


class StreamFlow(Workflow):
    """Streaming RL flow with hooks for automatic scorer batch-size tuning.

    The launch loop is intentionally not implemented yet — this scaffold
    establishes the contract that the controller and the streaming
    generator/scorer tasks will plug into:

    * Scorer tasks read the live batch size from the shared workflow ddict
      via :meth:`read_scorer_batch_size` on every loop pass, so a controller
      can change it without restarting tasks.
    * Scorer tasks push :class:`ThroughputSample` records into
      :attr:`monitor` via the :meth:`sample_scorer` context manager; the
      controller consumes those samples.
    """

    BATCH_SIZE_KEY = "stream_scorer_batch_size"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        trainer: Trainer,
        evaluate_func: Callable,
        asyncflow: Any,
        flow_config: StreamFlowConfig,
    ):
        super().__init__(
            model_config=model_config,
            trainer=trainer,
            evaluate_func=evaluate_func,
            asyncflow=asyncflow,
        )
        self.flow_config = flow_config
        self.monitor = ThroughputMonitor(window_size=flow_config.tuning.window_size)
        self._scorer_tasks: list = []
        self._generator_tasks: list = []

    def initial_scorer_batch_size(self) -> int:
        """Resolve the batch size to start the loop with.

        When tuning is enabled, the tuner's ``initial_batch_size`` wins (it
        is what the controller's first arm pull will use). Otherwise the
        static ``scorer_batch_size`` from the flow config is used.
        """
        tuning = self.flow_config.tuning
        if tuning.enabled:
            return tuning.initial_batch_size
        return self.flow_config.scorer_batch_size

    @classmethod
    def read_scorer_batch_size(cls, workflow_ddict: Any, default: int) -> int:
        """Read the current scorer batch size from the shared ddict.

        Tasks call this once per loop iteration so a controller running in a
        separate coroutine can adjust the batch size live. Returns
        ``default`` when the key is missing (cold start) or unparseable.
        """
        try:
            value = workflow_ddict[cls.BATCH_SIZE_KEY]
        except (KeyError, TypeError):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def publish_scorer_batch_size(self, workflow_ddict: Any, batch_size: int) -> None:
        """Publish a new scorer batch size for tasks to pick up on their
        next loop pass. Validated against the tuning bounds when enabled."""
        bs = int(batch_size)
        tuning = self.flow_config.tuning
        if tuning.enabled and not (
            tuning.min_batch_size <= bs <= tuning.max_batch_size
        ):
            raise ValueError(
                f"batch_size {bs} outside tuning bounds "
                f"[{tuning.min_batch_size}, {tuning.max_batch_size}]"
            )
        workflow_ddict[self.BATCH_SIZE_KEY] = bs

    def sample_scorer(
        self, batch_size: int, items: int, wait_s: float = 0.0
    ) -> "_SampleRecorder":
        """Context manager that times a scorer batch and records a
        :class:`ThroughputSample` on exit. Use as::

            with flow.sample_scorer(batch_size, len(batch), wait_s=wait):
                run_scoring(batch)
        """
        return _SampleRecorder(
            monitor=self.monitor,
            task="scorer",
            batch_size=batch_size,
            items=items,
            wait_s=wait_s,
        )

    async def launch(self, iterations: Optional[int] = None) -> None:
        """Streaming launch loop. Not implemented yet.

        Lands in a follow-up alongside the UCB controller; this scaffold
        ships the configuration surface, the monitor, and the dynamic
        batch-size hooks the controller and streaming tasks will use.
        """
        raise NotImplementedError(
            "StreamFlow.launch is not implemented yet; this scaffold ships "
            "the batch-tuning hooks (StreamFlow.read_scorer_batch_size, "
            "StreamFlow.publish_scorer_batch_size, StreamFlow.sample_scorer, "
            "StreamFlow.monitor) that the streaming loop and controller "
            "will use."
        )


class _SampleRecorder:
    """Context manager returned by :meth:`StreamFlow.sample_scorer`."""

    def __init__(
        self,
        *,
        monitor: ThroughputMonitor,
        task: str,
        batch_size: int,
        items: int,
        wait_s: float,
    ):
        self._monitor = monitor
        self._task = task
        self._batch_size = batch_size
        self._items = items
        self._wait_s = wait_s
        self._start: Optional[float] = None

    def __enter__(self) -> "_SampleRecorder":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = time.perf_counter() - (self._start or time.perf_counter())
        self._monitor.record(
            ThroughputSample(
                batch_size=self._batch_size,
                items=self._items,
                duration_s=duration,
                wait_s=self._wait_s,
                task=self._task,
            )
        )
        return False
