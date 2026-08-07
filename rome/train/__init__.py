"""Training algorithms available to ROME-A's training manager.

Adding one is a single task: subclass :class:`~rome.train.base.TrainTask`,
implement ``train(dataset, output_dir) -> checkpoint_path``, and hand the
instance to a :class:`~rome.trainer.TrainerConfig`. The trainers here are
imported lazily so a protein workflow never pays for TRL, and an LLM workflow
never needs pyarrow.
"""

from rome.train.base import FunctionTrainer, TrainTask

__all__ = [
    "TrainTask",
    "FunctionTrainer",
    "GRPOTrainer",
    "GRPOConfig",
    "ModelConfig",
    "ProteinMPNNTrainer",
    "ProteinMPNNConfig",
]

_LAZY = {
    "GRPOTrainer": "rome.train.llm",
    "GRPOConfig": "rome.train.llm",
    "ModelConfig": "rome.train.llm",
    "ProteinMPNNTrainer": "rome.train.mpnn",
    "ProteinMPNNConfig": "rome.train.mpnn",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
