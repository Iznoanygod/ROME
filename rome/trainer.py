"""Training Manager — creates and schedules training tasks, publishes checkpoints.

Responsibilities, straight off the design slide:

* create and schedule training tasks on HPC — every round is submitted to the
  ``radical.asyncflow`` engine the host workflow handed to ROME-A, so the
  training task gets its own nodes and GPUs like any other workflow task;
* publish updated checkpoints back to the workflow, so the stream manager and
  the host workflow pick up the improved model mid-campaign;
* answer whether training is *possible*, *running*, or *finished*.

Training starts automatically once enough data accumulates (the data manager's
``min_samples`` threshold) or is triggered manually by the workflow awaiting
:meth:`Trainer.train`.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dragon.native.event import Event

from rome.data import DataManager
from rome.train.base import FunctionTrainer, TrainTask
from rome.utils import (
    MODEL_PATH_KEY,
    MODEL_VERSION_KEY,
    Namespace,
    resource_description,
    submit_task,
)


class TrainerStatus(Enum):
    """Status of the ROME-A training manager.

    ``NOT_ENOUGH_DATA`` and ``WAITING`` are both idle, but they answer different
    questions: the first says a round is *not currently possible*, the second
    says it is possible and about to start.
    """

    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4
    NOT_ENOUGH_DATA = 5
    TRAINING_COMPLETE = 6
    WAITING = 7
    FAILED = 8

    @property
    def is_idle(self) -> bool:
        return self in (
            TrainerStatus.NOT_ENOUGH_DATA,
            TrainerStatus.WAITING,
            TrainerStatus.TRAINING_COMPLETE,
        )

    @property
    def is_terminal(self) -> bool:
        return self in (TrainerStatus.STOPPED, TrainerStatus.FAILED)


@dataclass
class TrainerConfig:
    """Configuration for the training manager.

    Parameters
    ----------
    trainer : TrainTask or Callable
        The training algorithm. A bare callable
        ``(dataset, output_dir, **kwargs) -> checkpoint_path`` is wrapped in a
        :class:`~rome.train.base.FunctionTrainer` automatically.
    checkpoint_dir : str
        Root for published checkpoints; rounds land in
        ``<checkpoint_dir>/<trainer name>/v<version>``.
    auto_train : bool
        Poll the data manager and fire a round as soon as enough data has
        accumulated. Turn off to drive training purely by hand.
    max_rounds : Optional[int]
        Stop after this many completed rounds. ``None`` runs until stopped.
    poll_interval : float
        Seconds between data-threshold checks in the auto-train loop.
    task_description : Optional[dict]
        Backend-specific resource request forwarded to asyncflow. When
        ``None``, one is derived from the trainer's ``gpus``/``nodes``.
    train_kwargs : dict
        Extra keyword arguments forwarded to every ``TrainTask.train`` call.
    on_checkpoint : Optional[Callable[[str, int], None]]
        Called with ``(checkpoint_path, version)`` after each successful round.
        The manager registers the stream manager's hot-swap hook here.
    stop_on_failure : bool
        Move to ``FAILED`` and end the auto-train loop when a round raises.
        When ``False`` (default) the failure is recorded and polling continues.
    """

    trainer: Any = None
    checkpoint_dir: str = "./rome_checkpoints"
    auto_train: bool = True
    max_rounds: Optional[int] = None
    poll_interval: float = 5.0
    task_description: Optional[Dict[str, Any]] = None
    train_kwargs: Dict[str, Any] = field(default_factory=dict)
    on_checkpoint: Optional[Callable[[str, int], None]] = None
    stop_on_failure: bool = False


class Trainer:
    """Training manager in the ROME framework.

    Parameters
    ----------
    ddict : Namespace
        Shared state — the same DDict view every other ROME-A component uses.
    data : DataManager
        Corpus the training rounds draw from.
    asyncflow : WorkflowEngine
        Engine training tasks are submitted to.
    config : TrainerConfig, optional
        May also be supplied later to :meth:`start`.
    """

    def __init__(
        self,
        ddict: Namespace,
        data: DataManager,
        asyncflow: Any,
        config: Optional[TrainerConfig] = None,
    ):
        self.ddict = ddict
        self.data = data
        self.asyncflow = asyncflow
        self.config = config or TrainerConfig()
        self.stop_event = Event()
        self.listener_fut: Optional[asyncio.Task] = None

        self._status = TrainerStatus.NOT_STARTED
        self._rounds_completed = 0
        self._last_error: Optional[str] = None
        self._in_flight = False
        #: Future for the round currently in flight, for diagnostics only.
        self._round_fut = None
        self._extra_callbacks: List[Callable[[str, int], None]] = []

    # -- lifecycle ----------------------------------------------------------

    async def start(self, config: Optional[TrainerConfig] = None) -> Optional[asyncio.Task]:
        """Start the training manager.

        With ``auto_train`` on this spawns the polling loop that fires a round
        whenever the corpus crosses ``min_samples``. With it off the manager
        simply becomes available for manual :meth:`train` calls.

        The poll loop is a plain asyncio task in the manager's own process —
        only the training rounds themselves go to the execution backend.
        """
        if config is not None:
            self.config = config
        if self.config.trainer is None:
            raise ValueError("TrainerConfig.trainer must be set before start()")

        self._status = TrainerStatus.STARTING
        self.stop_event.clear()
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Settle on a real status before returning: a caller that checks
        # get_training_status() immediately after start() should see whether a
        # round is possible, not that the loop has not been scheduled yet.
        self._status = self._idle_status()
        if not self.config.auto_train:
            return None

        self.listener_fut = asyncio.create_task(self._trainer_listener())
        return self.listener_fut

    async def _trainer_listener(self) -> None:
        """Poll the corpus and fire a training round when one becomes possible."""
        self._status = self._idle_status()
        while not self.stop_event.is_set():
            if self._rounds_exhausted():
                self._status = TrainerStatus.TRAINING_COMPLETE
                return
            if self._in_flight or not self.data.ready_to_train():
                self._status = self._idle_status()
                await asyncio.sleep(self.config.poll_interval)
                continue
            try:
                await self._run_round()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced via status
                self._record_failure(exc)
                if self.config.stop_on_failure:
                    return
                await asyncio.sleep(self.config.poll_interval)
        self._status = TrainerStatus.STOPPED

    async def stop(self, wait_for_stop: bool = True, timeout: float = 300.0) -> None:
        """Stop the training manager.

        An in-flight round is *not* cancelled — killing a half-finished
        fine-tune would leave a torn checkpoint — so this waits it out instead.
        """
        self._status = TrainerStatus.STOPPING
        self.stop_event.set()
        if not wait_for_stop:
            return
        deadline = time.time() + timeout
        while self._in_flight and time.time() < deadline:
            await asyncio.sleep(0.05)
        if self.listener_fut is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.listener_fut),
                    timeout=max(0.0, deadline - time.time()) or 0.1,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._status is not TrainerStatus.FAILED:
            self._status = TrainerStatus.STOPPED

    # -- training -----------------------------------------------------------

    async def train(self, force: bool = False, **kwargs: Any) -> Optional[str]:
        """Run one training round now — the workflow's manual trigger.

        Returns the path to the new checkpoint, or ``None`` when the round was
        skipped because the corpus was not ready (pass ``force=True`` to train
        anyway) or another round is already in flight.

        Failures propagate: a manual call is expected to surface its own
        errors, unlike the auto-train loop which records them in
        :attr:`status`.
        """
        if not force and not self.data.ready_to_train():
            self._status = TrainerStatus.NOT_ENOUGH_DATA
            return None
        return await self._run_round(**kwargs)

    async def _run_round(self, **kwargs: Any) -> Optional[str]:
        if self._in_flight:
            return None
        self._in_flight = True
        try:
            return await self._execute_round(**kwargs)
        finally:
            self._in_flight = False

    async def _execute_round(self, **kwargs: Any) -> str:
        """Build the dataset, submit the training task, publish the checkpoint."""
        task = self.train_task
        self._status = TrainerStatus.RUNNING

        dataset = self.data.get_dataset(as_hf_dataset=task.wants_hf_dataset)
        task.validate(dataset)
        sample_count = self._dataset_size(dataset)

        version = self.model_version + 1
        output_dir = task.prepare_output_dir(
            self.config.checkpoint_dir, version, task.name
        )

        call_kwargs = dict(self.config.train_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("model_version", version)

        checkpoint = await self._submit_round(task, dataset, output_dir, call_kwargs)
        self._publish(checkpoint or output_dir, version, sample_count)
        return checkpoint or output_dir

    async def _submit_round(self, task, dataset, output_dir, call_kwargs) -> Any:
        """Hand one training round to asyncflow.

        ``TrainTask.train`` is synchronous and blocking on purpose — a
        fine-tune should not have to be written as a coroutine — so the task
        body runs it in a thread and awaits that. Resource requirements come
        from the ``TrainTask`` itself, which is what keeps "adding a new
        training algorithm requires just one task" true.
        """
        description = self.config.task_description
        if description is None:
            description = resource_description(gpus=task.gpus, nodes=task.nodes)

        async def train_entry():
            return await asyncio.to_thread(task.train, dataset, output_dir, **call_kwargs)

        train_entry.__name__ = f"rome_train_{task.name}"
        # Kept only so a stalled round can be inspected: a round that never
        # returns leaves no other handle on the submission.
        self._round_fut = submit_task(
            self.asyncflow, train_entry, task_description=description
        )
        return await self._round_fut

    def _publish(self, checkpoint: str, version: int, sample_count: int) -> None:
        """Make a finished checkpoint visible to the rest of the workflow.

        Order matters: the path is written *before* the version is bumped, so a
        stream task that notices the new version always finds a valid path
        behind it.
        """
        self.ddict[MODEL_PATH_KEY] = checkpoint
        self.ddict[MODEL_VERSION_KEY] = version
        self.ddict["last_trained_at"] = time.time()
        self.ddict["last_train_samples"] = sample_count
        self.data.mark_consumed()
        self._rounds_completed += 1
        self._status = (
            TrainerStatus.TRAINING_COMPLETE
            if self._rounds_exhausted()
            else self._idle_status()
        )
        for callback in self._checkpoint_callbacks():
            try:
                callback(checkpoint, version)
            except Exception:  # noqa: BLE001 - a bad hook must not lose a checkpoint
                traceback.print_exc()

    def _checkpoint_callbacks(self) -> List[Callable[[str, int], None]]:
        callbacks = list(self._extra_callbacks)
        if self.config.on_checkpoint is not None:
            callbacks.append(self.config.on_checkpoint)
        return callbacks

    def on_checkpoint(self, callback: Callable[[str, int], None]) -> Callable:
        """Register an extra ``(checkpoint_path, version)`` callback."""
        self._extra_callbacks.append(callback)
        return callback

    # -- status -------------------------------------------------------------

    @property
    def status(self) -> TrainerStatus:
        """Whether training is possible, running, or finished."""
        if self._in_flight or self._status is TrainerStatus.RUNNING:
            return TrainerStatus.RUNNING
        if self._status.is_terminal or self._status is TrainerStatus.TRAINING_COMPLETE:
            return self._status
        if self._status in (
            TrainerStatus.NOT_STARTED,
            TrainerStatus.STARTING,
            TrainerStatus.STOPPING,
        ):
            return self._status
        return self._idle_status()

    def get_status(self) -> TrainerStatus:
        """Alias for :attr:`status` — the API-call form from the design slide."""
        return self.status

    def _idle_status(self) -> TrainerStatus:
        return (
            TrainerStatus.WAITING
            if self.data.ready_to_train()
            else TrainerStatus.NOT_ENOUGH_DATA
        )

    def get_current_model(self) -> Optional[str]:
        """Path to the newest published checkpoint, or ``None`` before round 1."""
        return self.ddict.get(MODEL_PATH_KEY)

    @property
    def model_version(self) -> int:
        """Version of the newest published checkpoint (0 = never trained)."""
        return int(self.ddict.get(MODEL_VERSION_KEY, 0))

    @property
    def rounds_completed(self) -> int:
        return self._rounds_completed

    @property
    def last_error(self) -> Optional[str]:
        """Traceback of the most recent failed round, if any."""
        return self._last_error

    def report(self) -> Dict[str, Any]:
        """One-call summary for dashboards and logs."""
        return {
            "status": self.status.name,
            "model_version": self.model_version,
            "model_path": self.get_current_model(),
            "rounds_completed": self._rounds_completed,
            "corpus_size": self.data.total_count,
            "unconsumed": self.data.unconsumed_count,
            "last_error": self._last_error,
        }

    # -- helpers ------------------------------------------------------------

    @property
    def train_task(self) -> TrainTask:
        """The configured trainer, wrapping a bare callable if needed."""
        trainer = self.config.trainer
        if trainer is None:
            raise ValueError("TrainerConfig.trainer must be set")
        if isinstance(trainer, TrainTask):
            return trainer
        if callable(trainer):
            wrapped = FunctionTrainer(trainer)
            self.config.trainer = wrapped
            return wrapped
        raise TypeError(
            f"TrainerConfig.trainer must be a TrainTask or callable, "
            f"got {type(trainer).__name__}"
        )

    def _rounds_exhausted(self) -> bool:
        cap = self.config.max_rounds
        return cap is not None and self._rounds_completed >= cap

    @staticmethod
    def _dataset_size(dataset: Any) -> int:
        try:
            return len(dataset)
        except TypeError:
            return -1

    def _record_failure(self, exc: BaseException) -> None:
        self._last_error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._status = (
            TrainerStatus.FAILED if self.config.stop_on_failure else self._idle_status()
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<Trainer status={self.status.name} version={self.model_version} "
            f"rounds={self._rounds_completed}>"
        )
