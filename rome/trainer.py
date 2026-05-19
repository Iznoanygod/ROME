from typing import Any, Callable, Dict, List, Optional
from rome.config import ModelConfig
from datasets import Dataset
class Trainer:
    """Abstract base class for ROME training algorithms.
    
    Parameters
    ----------
    gpus_per_node : int
        GPUs per node.
    seed : int
        Random seed forwarded to the underlying TRL trainer.
    reward_funcs : List[Callable]
        List of reward functions to use for training.
        Reward functions run inside the trainer unless marked with the @Workflow.reward_task decorator,
        in which case they are expected to be launched as tasks by the workflow
    """
    def __init__(
        self,
        *,
        gpus: int = 1,
        reward_funcs: List[Callable],
    ) -> None:
        self._gpus = gpus
        self._reward_funcs = reward_funcs or []

    @property
    def reward_funcs(self) -> List[Callable]:
        return self._reward_funcs


    def train(self, model_config: ModelConfig, dataset: Dataset, **kwargs):
        """Execute training for the mode.

        Uses model_config for loading the model and tokenizer, as well as generation
        parameters. reward_funcs are passed to the model trainer to run on the trainer task
        unless marked with the @Workflow.reward_task decorator, in which case the workflow
        is responsible for launching them as tasks and passing the results back to the trainer.
        
        Parameters
        ----------
        model_config : ModelConfig
            Model configuration for the model and tokenizer
        """
        raise NotImplementedError("Trainer.train() must be implemented by subclasses.")

