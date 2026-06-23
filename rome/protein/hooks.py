"""Pluggable task hooks for :class:`ProteinBindingFlow`.

The flow itself only knows how to *orchestrate* — when to call MPNN, when
to run AF2, when to extract metrics, when to fire a training round. The
actual science calls go through these hooks. Production wires up the real
implementations from :mod:`rome.protein.tasks`; tests pass in dummies.

Hook contracts
--------------

``mpnn_generator_loop(config, worker_index, workflow_ddict, terminate_event)``
    Long-running coroutine. Continuously samples ProteinMPNN under the
    current weights and writes ``SequenceRecord``-shaped dicts into
    ``workflow_ddict["mpnn_outputs"][backbone_id]``. Must respect
    ``terminate_event`` and ``model_version`` for hot weight reload.

``af2_predict(config, fasta_dir, fasta_filename, output_dir) -> str``
    Run AF2 on one FASTA. Returns the output directory.

``extract_metrics(config, pipeline_id, cycle, af_output_dir, csv_out_path)
    -> list[AF2Result]``
    Parse AF2 outputs into score rows.

``mpnn_train(config, parquet_shard_path, output_checkpoint_dir) -> str``
    Run one training round on a sampled shard. Returns the new checkpoint
    directory.

The flow never imports a concrete tool — all four cross the seam through
``TaskHooks``.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from rome.protein.schema import AF2Result


MpnnGeneratorLoop = Callable[[Any, int, Any, Any], Awaitable[None]]
Af2Predict = Callable[[Any, str, str, str], Awaitable[str]]
ExtractMetrics = Callable[[Any, str, int, str, str], Awaitable[List[AF2Result]]]
# Trainer receives already-sampled corpus entries (list of dicts) so dummies
# don't need to read/write parquet. The real implementation materializes the
# shard internally before invoking foundry.
MpnnTrain = Callable[[Any, list, str], Awaitable[str]]


@dataclass
class TaskHooks:
    mpnn_generator_loop: Optional[MpnnGeneratorLoop] = None
    af2_predict: Optional[Af2Predict] = None
    extract_metrics: Optional[ExtractMetrics] = None
    mpnn_train: Optional[MpnnTrain] = None

    def resolved(self) -> "TaskHooks":
        """Fill any unset hook with the production default from tasks.py."""
        from rome.protein import tasks  # local import to avoid cycles

        return TaskHooks(
            mpnn_generator_loop=self.mpnn_generator_loop or tasks.mpnn_generate_loop,
            af2_predict=self.af2_predict or tasks.af2_predict_task,
            extract_metrics=self.extract_metrics or tasks.extract_metrics_task,
            mpnn_train=self.mpnn_train or tasks.mpnn_train_task,
        )
