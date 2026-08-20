"""Base class every ROME-A training algorithm implements.

"Adding new models and training algorithms requires just one task" — this is
that task's interface. A trainer receives a dataset built by the data manager
and an output directory, does whatever it does, and returns the path to a fresh
checkpoint. The training manager takes it from there: publishing the checkpoint,
bumping the model version, and letting the stream manager hot-swap weights.

Trainers are plain synchronous callables by default. The training manager runs
them off the event loop (or hands them to ``asyncflow`` for placement on HPC),
so a trainer that blocks for an hour is fine.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple


class TrainTask:
    """Abstract base for ROME-A training algorithms.

    Subclasses implement :meth:`train`. Everything else — when to run, where
    the data comes from, who hears about the new checkpoint — belongs to the
    training manager.

    Parameters
    ----------
    gpus : int
        GPUs a single training task needs. The training manager forwards this
        to the workflow engine when submitting the task.
    nodes : int
        Nodes a single training task needs.
    name : str, optional
        Human-readable label used in status reporting and checkpoint paths.
        Defaults to the class name.
    """

    def __init__(self, *, gpus: int = 1, nodes: int = 1, name: Optional[str] = None):
        self.gpus = gpus
        self.nodes = nodes
        self.name = name or type(self).__name__

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Train on ``dataset`` and write a checkpoint under ``output_dir``.

        Parameters
        ----------
        dataset : Any
            Whatever :meth:`rome.data.DataManager.get_dataset` produced — a
            list of record dicts by default, or a ``datasets.Dataset`` when the
            trainer asked for one via :attr:`wants_hf_dataset`.
        output_dir : str
            Directory the checkpoint should be written to. Already created.
        **kwargs
            Forwarded from :meth:`rome.trainer.Trainer.train`; implementations
            should tolerate unknown keys.

        Returns
        -------
        str
            Path to the checkpoint the stream manager should load. Returning
            ``output_dir`` itself is the common case.
        """
        raise NotImplementedError("TrainTask.train() must be implemented by subclasses")

    def as_command(self, dataset: Any, output_dir: str,
                   **kwargs: Any) -> Optional[Tuple[str, str]]:
        """Optionally run this round as an executable task instead of a function.

        Return ``(shell_command, checkpoint_path)`` to have the training manager
        submit ``shell_command`` as an *executable* task — a subprocess on the
        allocation's resources, the way IMPRESS submits its wrapper scripts —
        rather than pickling :meth:`train` into a worker. ``checkpoint_path`` is
        where the command will write the checkpoint, since an executable task's
        return value is not the path.

        Return ``None`` (the default) to run :meth:`train` in-process as a
        function task. A GPU fine-tune should prefer the command form: the
        subprocess exits when the round ends, releasing its VRAM.

        Completion contract: the command MUST write a file named
        ``train_complete`` (``rome.trainer.TRAIN_COMPLETE_MARKER``) into its
        ``output_dir`` as its *final* action, after the checkpoint is safely on
        disk. The training manager polls for that marker to detect the round
        finished, because on Dragon a task can run to completion and never
        resolve its future (a running service blocks result delivery — see
        ``docs/dragon.md``). The marker, not the checkpoint, is the signal: with
        publish-in-place the checkpoint path already exists from the prior round.
        """
        return None

    #: Set on subclasses that want a ``datasets.Dataset`` rather than a list.
    wants_hf_dataset: bool = False

    def validate(self, dataset: Any) -> None:
        """Raise if ``dataset`` cannot be trained on.

        Called by the training manager *before* a task is scheduled, so an
        unusable corpus fails fast on the manager instead of burning an HPC
        allocation. The default check is just "not empty".
        """
        try:
            size = len(dataset)
        except TypeError:
            return
        if size == 0:
            raise ValueError(f"{self.name}: refusing to train on an empty dataset")

    @staticmethod
    def prepare_output_dir(base_dir: str, version: int, name: str) -> str:
        """Standard checkpoint layout: ``<base_dir>/<name>/v<version>``."""
        path = os.path.join(base_dir, name, f"v{version}")
        os.makedirs(path, exist_ok=True)
        return path

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} name={self.name!r} gpus={self.gpus}>"


class FunctionTrainer(TrainTask):
    """Adapter that turns a bare function into a :class:`TrainTask`.

    The cheapest possible way to plug an existing training script into
    ROME-A::

        rome = Manager(trainer=FunctionTrainer(my_finetune, gpus=4))

    where ``my_finetune(dataset, output_dir, **kwargs) -> checkpoint_path``.
    """

    def __init__(self, func, *, gpus: int = 1, nodes: int = 1, name: Optional[str] = None,
                 wants_hf_dataset: bool = False):
        super().__init__(gpus=gpus, nodes=nodes, name=name or getattr(func, "__name__", "function"))
        self._func = func
        self.wants_hf_dataset = wants_hf_dataset

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        result = self._func(dataset, output_dir, **kwargs)
        # A training function that returns nothing is assumed to have written
        # its checkpoint where it was told to.
        return result if isinstance(result, str) else output_dir


__all__: List[str] = ["TrainTask", "FunctionTrainer"]
