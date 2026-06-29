"""Pluggable task hooks for :class:`ProteinBindingFlow`.

The flow itself only knows how to *orchestrate* — when to call MPNN, when
to run the structure predictor, when to extract metrics, when to fire a
training round. The actual science calls go through these hooks.
Production wires up the real implementations from :mod:`rome.protein.tasks`;
tests pass in dummies.

Hook contracts (mirroring the IMPRESS main-branch AF2 pipeline)
---------------------------------------------------------------

``mpnn_generator_loop(config, worker_index, workflow_ddict, terminate_event)``
    Long-running coroutine. Continuously samples ProteinMPNN under the
    current weights and writes ``SequenceRecord``-shaped dicts into
    ``workflow_ddict["mpnn_outputs"][backbone_id]``. Must respect
    ``terminate_event`` and ``model_version`` for hot weight reload.

``predict_structure(config, fasta_dir, fasta_filename, output_dir) -> str``
    Run the structure predictor on a single FASTA. Matches the IMPRESS
    main-branch ``af2_multimer_reduced.sh`` three-arg signature. Returns
    the prediction output directory.

``stage_prediction(config, prediction_output_dir, target_fasta, backbone_id) -> str``
    *Optional.* Post-process the predictor's outputs (file renames, chain
    rewrites, output flattening). Default = no-op; AF2 writes outputs
    directly into a form the extractor accepts so no staging is needed.
    Required when running a non-AF2 predictor (e.g. Boltz, which emits
    multi-char PDB chain IDs and a nested output tree).

``extract_metrics(config, pipeline_id, cycle, af_output_dir, csv_out_path)
    -> list[PredictionResult]``
    Parse the predictor's outputs into score rows. Wraps the main-branch
    ``plddt_extract_pipeline.py``.

``mpnn_train(config, sampled_entries, output_checkpoint_dir) -> str``
    Run one training round on a sampled shard. Returns the new
    checkpoint directory. ROME-specific extension — not part of the
    IMPRESS pipeline.

The flow never imports a concrete tool — all five cross the seam through
``TaskHooks``.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from rome.protein.schema import PredictionResult


MpnnGeneratorLoop = Callable[[Any, int, Any, Any], Awaitable[None]]
# (config, fasta_dir, fasta_filename, output_dir) -> output_dir
PredictStructure = Callable[[Any, str, str, str], Awaitable[str]]
# (config, prediction_output_dir, target_fasta, backbone_id) -> staged_dir
StagePrediction = Callable[[Any, str, str, str], Awaitable[str]]
# (config, pipeline_id, cycle, af_output_dir, csv_out_path) -> list[PredictionResult]
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
