"""Pluggable task hooks for :class:`ProteinBindingFlow`.

The flow itself only knows how to *orchestrate* — when to call MPNN, when
to run the structure predictor, when to stage outputs, when to extract
metrics, when to fire a training round. The actual science calls go
through these hooks. Production wires up the real implementations from
:mod:`rome.protein.tasks`; tests pass in dummies.

Hook contracts
--------------

``mpnn_generator_loop(config, worker_index, workflow_ddict, terminate_event)``
    Long-running coroutine. Continuously samples ProteinMPNN under the
    current weights and writes ``SequenceRecord``-shaped dicts into
    ``workflow_ddict["mpnn_outputs"][backbone_id]``. Must respect
    ``terminate_event`` and ``model_version`` for hot weight reload.
    Mirrors IMPRESS's **s1** stage.

``predict_structure(config, fasta_path, output_dir) -> str``
    Run the structure predictor (Boltz by default, AF2 alternate) on a
    single paired FASTA. Returns the prediction output directory.
    Mirrors IMPRESS's **s4** stage.

``stage_prediction(config, prediction_output_dir, best_model_dst,
                    best_ptm_dst, backbone_id) -> str``
    Stage the best model + confidence JSON from the predictor's nested
    output layout into canonical locations, renaming multi-char Boltz
    chains (``pdz`` -> ``A``, ``pep`` -> ``B``) so downstream PyRosetta /
    next-pass MPNN can read them. Returns the staged PDB path.
    Mirrors IMPRESS's **s4_post_exec** stage.

``extract_metrics(config, pipeline_id, cycle, prediction_root, csv_out_path)
    -> list[PredictionResult]``
    Parse staged PDB + JSON outputs into score rows. ``prediction_root``
    is the staging root (parent of ``best_models/`` and ``best_ptm/``).
    Mirrors IMPRESS's **s5** stage.

``mpnn_train(config, sampled_entries, output_checkpoint_dir) -> str``
    Run one training round on a sampled shard of corpus entries.
    Returns the new checkpoint directory. ROME-specific extension - not
    part of the IMPRESS pipeline.

The flow never imports a concrete tool - all five cross the seam through
``TaskHooks``.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from rome.protein.schema import PredictionResult


MpnnGeneratorLoop = Callable[[Any, int, Any, Any], Awaitable[None]]
PredictStructure = Callable[[Any, str, str], Awaitable[str]]
StagePrediction = Callable[[Any, str, str, str, str], Awaitable[str]]
ExtractMetrics = Callable[[Any, str, int, str, str], Awaitable[List[PredictionResult]]]
# Trainer receives already-sampled corpus entries (list of dicts) so dummies
# don't need to read/write parquet. The real implementation materializes the
# shard internally before invoking foundry.
MpnnTrain = Callable[[Any, list, str], Awaitable[str]]


@dataclass
class TaskHooks:
    mpnn_generator_loop: Optional[MpnnGeneratorLoop] = None
    predict_structure: Optional[PredictStructure] = None
    stage_prediction: Optional[StagePrediction] = None
    extract_metrics: Optional[ExtractMetrics] = None
    mpnn_train: Optional[MpnnTrain] = None

    def resolved(self) -> "TaskHooks":
        """Fill any unset hook with the production default from tasks.py."""
        from rome.protein import tasks  # local import to avoid cycles

        return TaskHooks(
            mpnn_generator_loop=self.mpnn_generator_loop or tasks.mpnn_generate_loop,
            predict_structure=self.predict_structure or tasks.predict_structure_task,
            stage_prediction=self.stage_prediction or tasks.stage_prediction_task,
            extract_metrics=self.extract_metrics or tasks.extract_metrics_task,
            mpnn_train=self.mpnn_train or tasks.mpnn_train_task,
        )
