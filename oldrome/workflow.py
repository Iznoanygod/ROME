from typing import Any, Callable

from radical.asyncflow import WorkflowEngine

from rome.config import ModelConfig
from rome.trainer import Trainer

class Workflow():
    """Abstract base class for ROME flows.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration for the model and tokenizer
    trainer : Trainer
        Training algorithm (e.g. ``GRPO``, ``SFT``).
    evaluate_func : Callable, optional
        Per-iteration evaluation function. Plain -> run inline; 
        decorated with ``@Workflow.evaluate_task`` -> run as a task 
        Returns a scalar that drives the stop criterion when
        ``flow_config.reward_threshold`` is set.
    flow_config : SequentialFlowConfig, optional
        Workflow knobs (iterations, threshold, dataset, checkpointing, ...).
    asyncflow : WorkflowEngine, optional
        The workflow engine instance used to manage async tasks.
    """
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

    # function decorators
    
    # need a decorator for reward functions that marks reward function as task
    # workflow will have to implement the reward task launching
    @staticmethod
    def reward_task(func):
        """Mark a function as a task function.

        Tagged functions are dispatched to workflow backend instead of being
        called inline. They share the same signature as local reward funcs:
        ``fn(prompts, completions, ground_truth, **kwargs) -> list[float]``.
        """
        func._is_reward_task = True
        return func

    @staticmethod
    def evaluate_task(func):
        """Mark a function as a function task.

        Returns a scalar score per iteration.
        """
        func._is_evaluate_task = True
        return func

    