from typing import Callable, List, Optional, Dict, Any
import typeguard

class Trainer:
    """
    Parameters
    ----------
    gpus_per_node : int
        GPUs per node.
    seed : int
        Random seed forwarded to the underlying TRL trainer.
    """
    def __init__(
        self,
        *,
        gpus: int = 1,
        dataset,
        reward_funcs: List[Callable],
    ) -> None:
        self._gpus = gpus
        self._dataset = dataset
        self._reward_funcs = reward_funcs or []

    def train(self, model_config: ModelConfig, **kwargs):
        """Train the model for one training step."""
        raise NotImplementedError("Trainer.train() must be implemented by subclasses.")

