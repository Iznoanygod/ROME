from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from oldrome.protein.schema import PredictionResult


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
        from oldrome.protein import tasks  # local import to avoid cycles

        return TaskHooks(
            mpnn_generator_loop=self.mpnn_generator_loop or tasks.mpnn_generate_loop,
            predict_structure=self.predict_structure or tasks.predict_structure_task,
            stage_prediction=self.stage_prediction or tasks.stage_prediction_task,
            extract_metrics=self.extract_metrics or tasks.extract_metrics_task,
            mpnn_train=self.mpnn_train or tasks.mpnn_train_task,
        )