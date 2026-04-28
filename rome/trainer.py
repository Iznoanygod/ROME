from typing import Callable, List, Optional, Dict, Any
import typeguard

class Trainer:
    @typeguard.typechecked
    def __init__(
        self,
        required_gpus: int = 1,
        training_kwargs: Optional[Dict[str, Any]] = None,
        reward_funcs: Optional[List[Callable]] = None,
    ):
        self.training_kwargs = training_kwargs or {}
        self.reward_funcs = reward_funcs or []
    
    def run_training(self, *args, **kwargs):
        raise NotImplementedError("Trainer subclasses must implement run_training()")
    