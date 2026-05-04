from radical.asyncflow import WorkflowEngine

from rome.config import ModelConfig, SequentialFlowConfig
from rome.trainer import Trainer, TrainerResult

class Workflow():
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        trainer: Trainer,
        evaluate_func: Callable,
        asyncflow: WorkflowEngine,
        
    ):
        self.model_config = model_config
        self.trainer = trainer
        self.evaluate_func = evaluate_func
        self.asyncflow = asyncflow

    async def launch(self, **kwargs) -> Any:
        """Launch the workflow. Must be implemented by subclasses."""
        raise NotImplementedError("Workflow.launch() must be implemented by subclasses.")

    