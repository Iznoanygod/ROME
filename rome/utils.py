"""Shared helpers for laying ROME-A state out inside a Dragon DDict.

Every ROME-A component reads and writes the same distributed dictionary, which
is what lets a task on one node add training data that a training task on
another node picks up. The one rule that makes that safe is:

    **never read-modify-write a shared container.**

``d = ddict["corpus"]; d[uid] = rec; ddict["corpus"] = d`` loses records the
moment two nodes do it at once. So ROME-A gives every record, request and
result its own key and rebuilds collections by scanning a key prefix — single
key writes are atomic in a DDict, so nothing is ever clobbered.

:class:`Namespace` is the small amount of glue that makes that layout pleasant
to work with. It is a *view*, not a store: it holds no state of its own and
several views over the same DDict compose freely.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Tuple

#: Separator between namespace segments. Unlikely to appear in a uid or a name.
SEP = "|"

#: Key holding the checkpoint the workflow should currently be using.
MODEL_PATH_KEY = "model_path"
#: Key holding the monotonically increasing checkpoint version. Stream tasks
#: poll this to decide when to hot-swap weights.
MODEL_VERSION_KEY = "model_version"

_MISSING = object()

#: Per-thread Dragon client handles, keyed by the dictionary's serialized form.
_thread_local = threading.local()


def thread_handle(ddict: MutableMapping, serialized: Optional[str]) -> MutableMapping:
    """Return a DDict client handle owned by the calling thread.

    A Dragon DDict handle multiplexes an FLI channel and is **not** safe to
    share between threads: concurrent use corrupts the channel and surfaces as
    ``dragon.fli.DragonFLIEOT``, usually from an unrelated later operation. It
    is not a hypothetical — ROME-A runs the manager's event loop and its
    ``asyncio.to_thread`` stream workers in one process against one dictionary,
    which reproduces it within a few hundred operations.

    Dragon's own remedy is that each consumer attaches its own handle, so that
    is what this does, once per thread and cached. Attaching is cheap and the
    thread pool is bounded, so the cache stays small.

    ``serialized is None`` means the backing is a plain mapping (the unit-test
    and single-process case), which is returned unchanged.
    """
    if serialized is None:
        return ddict
    cache = getattr(_thread_local, "handles", None)
    if cache is None:
        cache = _thread_local.handles = {}
    handle = cache.get(serialized)
    if handle is None:
        handle = type(ddict).attach(serialized)
        cache[serialized] = handle
    return handle


class Namespace(MutableMapping):
    """A prefixed view over a Dragon ``DDict`` (or any mapping).

    Every operation goes through a handle owned by the calling thread — see
    :func:`thread_handle` — so views may be shared freely across the manager
    and its stream workers.

    Parameters
    ----------
    ddict : MutableMapping
        The shared dictionary. Normally a ``dragon.data.ddict.DDict``; any
        mapping works, which is what makes the components unit-testable.
    prefix : str
        Prepended to every key this view touches.
    """

    __slots__ = ("_ddict", "_prefix", "_serialized")

    def __init__(self, ddict: MutableMapping, prefix: str = ""):
        self._ddict = ddict
        self._prefix = prefix
        # Serialized once, here, rather than per access: serialize() is itself
        # a client operation, and calling it from many threads would be the
        # very race this is meant to avoid.
        self._serialized = ddict.serialize() if hasattr(ddict, "serialize") else None

    @property
    def _target(self) -> MutableMapping:
        """The dictionary handle this thread may safely use."""
        return thread_handle(self._ddict, self._serialized)

    # -- composition --------------------------------------------------------

    @property
    def ddict(self) -> MutableMapping:
        """The underlying shared dictionary."""
        return self._ddict

    @property
    def prefix(self) -> str:
        return self._prefix

    def namespace(self, *segments: Any) -> "Namespace":
        """Derive a child view: ``ns.namespace("record")`` -> keys ``record|…``."""
        suffix = SEP.join(str(s) for s in segments)
        child = Namespace.__new__(Namespace)
        child._ddict = self._ddict
        child._prefix = f"{self._prefix}{suffix}{SEP}"
        # Inherited rather than recomputed: serialize() is a client operation,
        # and child views are created on hot paths.
        child._serialized = self._serialized
        return child

    def _full(self, key: str) -> str:
        return self._prefix + key

    # -- single-key access --------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._target[self._full(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._target[self._full(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._target[self._full(key)]

    def __contains__(self, key: object) -> bool:
        return self.get(str(key), _MISSING) is not _MISSING

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        """Read ``key``, returning ``default`` when absent.

        Goes through ``__getitem__`` rather than ``DDict.get`` so the behaviour
        is identical whichever mapping is underneath.
        """
        try:
            return self._target[self._full(key)]
        except (KeyError, TypeError):
            return default

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        """Remove ``key`` and return its value.

        This is ROME-A's claim protocol: whoever pops a key owns that item, so
        two stream replicas draining the same queue never process a request
        twice. Correctness rests on the underlying pop being atomic, which is
        why it delegates rather than doing read-then-delete — a Dragon DDict
        resolves the contention manager-side, and measurably: four threads
        racing over 120 keys claim 120 with zero duplicates.

        The single-argument form is deliberate. ``dict.pop(k, default)``
        returns the default while ``DDict.pop(k, default)`` raises
        ``DDictKeyError`` regardless, so asking for the default would behave
        differently on the two backings; catching ``KeyError`` (which
        ``DDictKeyError`` subclasses) behaves the same on both.
        """
        try:
            return self._target.pop(self._full(key))
        except (KeyError, TypeError):
            if default is _MISSING:
                raise KeyError(key)
            return default

    def increment(self, key: str, amount: int = 1) -> int:
        """Bump an integer counter and return the new value.

        Only safe when a single component owns the counter — ROME-A keeps to
        that (the training manager owns ``model_version``, and so on).
        """
        new = int(self.get(key, 0)) + amount
        self[key] = new
        return new

    # -- prefix scans -------------------------------------------------------

    def _raw_keys(self) -> List[str]:
        try:
            raw = list(self._target.keys())
        except (AttributeError, TypeError):  # pragma: no cover - exotic mapping
            return []
        return [k for k in raw if isinstance(k, str)]

    def keys(self, prefix: str = "") -> List[str]:  # type: ignore[override]
        """Namespace-relative keys, optionally filtered by a further prefix."""
        full_prefix = self._full(prefix)
        cut = len(self._prefix)
        return sorted(k[cut:] for k in self._raw_keys() if k.startswith(full_prefix))

    def items(self, prefix: str = "") -> List[Tuple[str, Any]]:  # type: ignore[override]
        """``(key, value)`` pairs under ``prefix``, skipping keys that vanish mid-scan."""
        out: List[Tuple[str, Any]] = []
        for key in self.keys(prefix):
            value = self.get(key, _MISSING)
            if value is not _MISSING:
                out.append((key, value))
        return out

    def values(self, prefix: str = "") -> List[Any]:  # type: ignore[override]
        return [v for _, v in self.items(prefix)]

    def drain(self, prefix: str = "", limit: Optional[int] = None) -> List[Tuple[str, Any]]:
        """Pop and return up to ``limit`` ``(key, value)`` pairs under ``prefix``."""
        out: List[Tuple[str, Any]] = []
        for key in self.keys(prefix):
            if limit is not None and len(out) >= limit:
                break
            value = self.pop(key, _MISSING)
            if value is not _MISSING:
                out.append((key, value))
        return out

    def clear(self, prefix: str = "") -> int:  # type: ignore[override]
        """Delete every key under ``prefix``; returns how many were removed."""
        return len(self.drain(prefix))

    def snapshot(self, prefix: str = "") -> Dict[str, Any]:
        """A plain ``dict`` copy of this view — handy when logging or debugging."""
        return dict(self.items(prefix))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Namespace prefix={self._prefix!r} keys={len(self.keys())}>"


# ---------------------------------------------------------------------------
# asyncflow task submission
# ---------------------------------------------------------------------------

def submit_task(
    asyncflow,
    func,
    *args,
    task_description: Optional[Dict[str, Any]] = None,
    service: bool = False,
    executable: bool = False,
    **kwargs,
):
    """Submit ``func`` to the workflow engine as a task.

    ROME-A schedules nothing itself — asyncflow is the execution backend, and
    this is the single place ROME-A talks to it.

    Parameters
    ----------
    asyncflow : WorkflowEngine
        The engine the host workflow handed to ROME-A.
    func : coroutine function
        The task body. asyncflow only accepts ``async def``. For a *function*
        task a blocking trainer or inference call belongs inside an
        ``asyncio.to_thread``; for an *executable* task ``func`` instead returns
        the shell command string to run (the same shape IMPRESS uses).
    task_description : dict, optional
        Backend-specific resource request (ranks, GPUs, placement policy, …).
        Passed straight through — ROME-A does not interpret it, because the
        keys depend on which execution backend the workflow is running.
    service : bool
        Mark the task as a long-running service. ROME-A's inference and reward
        streams are services; a training round is not. Ignored for executable
        tasks.
    executable : bool
        Submit as an executable (command) task rather than a function task.
        ``func``'s return value is then the command line to run on the backend,
        not a Python result. This is how a training round runs as its own
        process — see :meth:`rome.train.base.TrainTask.as_command`.

    Returns
    -------
    asyncio.Future
        Resolves when the task completes (to its return value for a function
        task; to the backend's execution result for an executable one).
    """
    if executable:
        decorated = asyncflow.executable_task()(func)
    else:
        decorated = asyncflow.function_task(service=service)(func)
    return decorated(*args, task_description=task_description or {}, **kwargs)


def resource_description(
    gpus: int = 0, nodes: int = 1, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build a default ``task_description`` from ROME-A's coarse resource fields.

    The concrete keys an execution backend understands are its own business, so
    this only fills in the two that the RADICAL backends share, and anything a
    workflow passes in ``extra`` wins. Workflows on an exotic backend should set
    ``task_description`` explicitly on the config and ignore ``gpus``/``nodes``.
    """
    desc: Dict[str, Any] = {}
    if nodes and nodes != 1:
        desc["ranks"] = nodes
    if gpus:
        desc["gpus_per_rank"] = gpus
    if extra:
        desc.update(extra)
    return desc
