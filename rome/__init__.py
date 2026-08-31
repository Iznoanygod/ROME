"""ROME: the RADICAL Optimizer for Model Enhancement.

ROME provides model improvement as a component you bolt onto a workflow you
already have, rather than as a workflow you have to adopt. Three managers, each
a configurable unit:

* :class:`~rome.data.DataManager` — gathers scored outputs from the workflow and
  builds them into a training dataset;
* :class:`~rome.trainer.Trainer` — schedules training tasks and publishes
  checkpoints back to the workflow;
* :class:`~rome.stream.Stream` — runs inference and reward as persistent
  asynchronous streams and hot-swaps their weights when a checkpoint lands.

:class:`~rome.manager.Manager` wires the three together, so adoption is a few
API calls::

    from radical.asyncflow import WorkflowEngine
    import rome

    flow = await WorkflowEngine.create(backend=backend)
    manager = rome.Manager(
        flow,
        data_config=rome.DataConfig(min_samples=64),
        trainer_config=rome.TrainerConfig(trainer=my_trainer),
        stream_configs=[rome.StreamConfig(name="infer", process_func=my_infer)],
    )
    await manager.start()
"""

from rome.data import DataConfig, DataManager
from rome.manager import Manager
from rome.stream import (
    Stream,
    StreamConfig,
    StreamContext,
    StreamKind,
    StreamStatus,
    StreamTask,
)
from rome.train.base import FunctionTrainer, TrainTask
from rome.trainer import Trainer, TrainerConfig, TrainerStatus
from rome.utils import MODEL_PATH_KEY, MODEL_VERSION_KEY, Namespace

__version__ = "0.0.1"

__all__ = [
    "Manager",
    # data
    "DataManager",
    "DataConfig",
    # training
    "Trainer",
    "TrainerConfig",
    "TrainerStatus",
    "TrainTask",
    "FunctionTrainer",
    # streams
    "Stream",
    "StreamConfig",
    "StreamContext",
    "StreamKind",
    "StreamStatus",
    "StreamTask",
    # shared state
    "Namespace",
    "MODEL_PATH_KEY",
    "MODEL_VERSION_KEY",
]
