from typing import Any, Callable, List, Optional
from rome.config import ModelConfig
from datasets import Dataset


class Trainer:
    """Abstract base class for ROME training algorithms.

    Parameters
    ----------
    gpus : int
        GPUs per node.
    reward_funcs : List[Callable]
        List of reward functions to use for training. Reward functions
        run inside the trainer unless marked with the
        ``@Workflow.reward_task`` decorator, in which case they are
        expected to be launched as tasks by the workflow.
    dataset : Dataset, optional
        Training dataset. Passed into the constructor rather than
        ``train()`` so :class:`SequentialFlow` can call
        ``await trainer.train(model_config, workflow_ddict=...)``
        without needing to plumb the dataset through the flow's
        registered update task.
    """

    def __init__(
        self,
        *,
        gpus: int = 1,
        reward_funcs: List[Callable],
        dataset: Optional[Dataset] = None,
    ) -> None:
        self._gpus = gpus
        self._reward_funcs = reward_funcs or []
        self._dataset = dataset

    @property
    def reward_funcs(self) -> List[Callable]:
        return self._reward_funcs

    @property
    def dataset(self) -> Optional[Dataset]:
        return self._dataset

    async def train(self, model_config: ModelConfig, **kwargs) -> Any:
        """Execute training for the model.

        Async so callers (e.g. :class:`SequentialFlow`) can await it
        without blocking the event loop; concrete implementations that
        wrap a synchronous underlying trainer should offload it with
        ``await asyncio.to_thread(...)``.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration for the model and tokenizer.
        **kwargs
            Implementation-specific extras (e.g. ``workflow_ddict``).
        """
        raise NotImplementedError("Trainer.train() must be implemented by subclasses.")
