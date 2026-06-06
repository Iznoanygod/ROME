"""Throughput monitoring and batch-size tuning hooks for ROME flows.

This module is the data-collection and configuration layer for automatic
batch-size determination. Flows push :class:`ThroughputSample` records into a
:class:`ThroughputMonitor` from every generator/scorer task; a future
:class:`BatchSizeController` (UCB bandit) reads those samples and proposes the
next batch size.

The controller itself is intentionally left as an interface in this scaffold
— shipping the monitor first lets us collect real samples before committing
to a tuning strategy.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, Iterable, List, Optional


@dataclass
class ThroughputSample:
    """One observation of throughput for a given batch size.

    Attributes
    ----------
    batch_size : int
        The batch size in effect for the observed work.
    items : int
        Number of items processed in the sample (e.g. generations scored).
    duration_s : float
        Wall-clock time spent processing the batch.
    wait_s : float
        Wall-clock time spent waiting for the batch to fill before
        processing started. Captures the "idle while filling" cost called out
        in the design notes.
    task : str
        Role tag — e.g. ``"scorer"`` or ``"generator"``. Lets a controller
        tune per-role even when samples share a monitor.
    timestamp : float
        Unix epoch seconds at sample creation. Defaults to ``time.time()``.
    """

    batch_size: int
    items: int
    duration_s: float
    wait_s: float = 0.0
    task: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def throughput(self) -> float:
        """Items per second of processing time. Excludes ``wait_s``."""
        if self.duration_s <= 0:
            return 0.0
        return self.items / self.duration_s

    @property
    def effective_throughput(self) -> float:
        """Items per second including the fill wait — this is the signal the
        end-to-end loop-throughput objective ultimately cares about."""
        total = self.duration_s + self.wait_s
        if total <= 0:
            return 0.0
        return self.items / total


class ThroughputMonitor:
    """Bounded rolling-window collector of throughput samples.

    Tasks call :meth:`record` from inside their inner loop; the controller
    (or tests) call :meth:`samples` / :meth:`mean_throughput` to read back.
    Samples are partitioned by ``task`` tag so per-role queries are cheap.

    Thread-safe via a single mutex — fine for the current scale (small
    number of tasks pushing intermittently).
    """

    def __init__(self, window_size: int = 64):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        self._samples: Dict[str, Deque[ThroughputSample]] = {}
        self._lock = Lock()

    @property
    def window_size(self) -> int:
        return self._window_size

    def record(self, sample: ThroughputSample) -> None:
        with self._lock:
            buf = self._samples.get(sample.task)
            if buf is None:
                buf = deque(maxlen=self._window_size)
                self._samples[sample.task] = buf
            buf.append(sample)

    def samples(self, task: Optional[str] = None) -> List[ThroughputSample]:
        with self._lock:
            if task is None:
                return [s for buf in self._samples.values() for s in buf]
            return list(self._samples.get(task, ()))

    def samples_for_batch(
        self, batch_size: int, task: Optional[str] = None
    ) -> List[ThroughputSample]:
        """All samples observed at a specific batch size. The UCB controller
        uses this to compute per-arm reward statistics."""
        return [s for s in self.samples(task) if s.batch_size == batch_size]

    def mean_throughput(self, task: Optional[str] = None) -> float:
        ss = self.samples(task)
        if not ss:
            return 0.0
        return sum(s.throughput for s in ss) / len(ss)

    def mean_effective_throughput(self, task: Optional[str] = None) -> float:
        ss = self.samples(task)
        if not ss:
            return 0.0
        return sum(s.effective_throughput for s in ss) / len(ss)

    def mean_wait(self, task: Optional[str] = None) -> float:
        ss = self.samples(task)
        if not ss:
            return 0.0
        return sum(s.wait_s for s in ss) / len(ss)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


@dataclass
class BatchTuningConfig:
    """Configuration for the automatic batch-size controller.

    The controller is not yet implemented; this dataclass locks in the API
    the UCB-style controller will consume so flows and tests can be wired
    up against the final shape today.

    Parameters
    ----------
    enabled : bool
        Master switch. When ``False`` the flow uses a static batch size and
        the monitor still records samples (useful for offline analysis).
    min_batch_size, max_batch_size : int
        Inclusive bounds on candidate batch sizes.
    initial_batch_size : int
        Batch size used before the controller has enough samples to pick.
    window_size : int
        Rolling-window length passed to :class:`ThroughputMonitor`.
    objective : str
        Which monitor signal the controller optimizes. ``"effective"`` uses
        :attr:`ThroughputSample.effective_throughput` (the end-to-end signal
        chosen for the first implementation). ``"processing"`` ignores the
        fill wait and is provided for diagnostics.
    exploration_c : float
        UCB1 exploration constant. The classic value is ``sqrt(2) ~= 1.41``.
    candidate_batch_sizes : list[int], optional
        Explicit arms for the bandit. When ``None``, arms are auto-generated
        as a geometric sequence between ``min_batch_size`` and
        ``max_batch_size``.
    min_observations_per_arm : int
        The controller will pull each arm at least this many times before it
        is allowed to settle on a winner. Prevents premature convergence.
    tuning_interval_s : float
        Minimum wall-clock seconds between controller decisions. Keeps the
        controller from thrashing on noisy short samples.
    """

    enabled: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 64
    initial_batch_size: int = 4
    window_size: int = 64
    objective: str = "effective"
    exploration_c: float = 1.41421356
    candidate_batch_sizes: Optional[List[int]] = None
    min_observations_per_arm: int = 3
    tuning_interval_s: float = 30.0

    def __post_init__(self) -> None:
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if not (self.min_batch_size <= self.initial_batch_size <= self.max_batch_size):
            raise ValueError(
                "initial_batch_size must lie within [min_batch_size, max_batch_size]"
            )
        if self.objective not in ("effective", "processing"):
            raise ValueError("objective must be 'effective' or 'processing'")

    def candidates(self) -> List[int]:
        """Resolved list of candidate batch sizes (arms) for the controller."""
        if self.candidate_batch_sizes is not None:
            return [int(c) for c in self.candidate_batch_sizes]
        out: List[int] = []
        n = self.min_batch_size
        while n < self.max_batch_size:
            out.append(n)
            n *= 2
        out.append(self.max_batch_size)
        # de-duplicate while preserving order (covers min==max and edge ratios)
        seen = set()
        unique: List[int] = []
        for n in out:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique


class BatchSizeController:
    """Interface for batch-size controllers.

    Concrete implementations (e.g. a UCB1 bandit) read recent samples from a
    :class:`ThroughputMonitor` and propose the next batch size. This
    scaffold ships the interface only; the first controller lands in a
    follow-up so we can review real monitor output before committing to a
    tuning policy.
    """

    def __init__(self, config: BatchTuningConfig, monitor: ThroughputMonitor):
        self.config = config
        self.monitor = monitor

    def propose(self, task: str, current: int) -> int:
        """Return the batch size the flow should use for ``task`` next."""
        raise NotImplementedError(
            "BatchSizeController is an interface — pick a concrete tuner."
        )
