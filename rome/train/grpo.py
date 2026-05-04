import os
import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
from rome.trainer import Trainer

from dragon.data.ddict import DDict

class GRPO(Trainer):
    """GRPO trainer. See ``GRPOTrainer`` for the underlying implementation of the GRPO algorithm, and ``GRPOConfig`` for training knobs.
    
    Parameters
    ----------
    gpus : int
        Number of GPUs to use for training.
    dataset : Dataset
        Dataset to use for training.
    reward_funcs : List[Callable]
        List of reward functions to use for training.
        Reward functions run inside the trainer unless marked with the @Workflow.reward_task decorator,
        in which case they are processed by the workflow
    trainer_callbacks : Optional[List[Any]]
        List of callbacks for the trainer.
    rollout_func : Optional[Callable]
        Rollout function for generating trajectories. 
        If not provided, the default rollout function is used.
    grpo_config : Optional[GRPOConfig]
        Configuration for the GRPO trainer.
        If not provided, ROME defaults are used.
    """
    def __init__(
        self,
        *,
        gpus: int = 1,
        dataset,
        reward_funcs: List[Callable],
        trainer_callbacks: Optional[List[Any]] = None,
        rollout_func: Optional[Callable] = None,
        grpo_config: Optional[GRPOConfig] = None,
    ):
        super().__init__(gpus=gpus, dataset=dataset, reward_funcs=reward_funcs)
        self._trainer_callbacks = trainer_callbacks
        self._rollout_func = rollout_func
        self._grpo_config = grpo_config
        if self._grpo_config is None:
            self._grpo_config = GRPOConfig(
                learning_rate=5e-6,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.01,
                warmup_ratio = 0.1,
                lr_scheduler_type = "cosine",
                optim = "adamw_8bit",
                logging_steps=1,

                # how many to process at once per gpu
                per_device_train_batch_size=4,

                # how many steps to accumulate
                gradient_accumulation_steps=16,

                # how many generations for each prompt
                num_generations=4,

                # how many prompts to process at once for generation (should be <= per_device_train_batch_size)
                generation_batch_size = 4,

                num_train_epochs=3,
            )
    
    # default rollout_func to use for GRPO when none is provided and using generator tasks
    def _default_rollout_func(prompts: list[str], trainer: GRPOTrainer, **kwargs) -> list[str]:
        import uuid
        # give each prompt a request_id, put in the workflow_ddict
        workflow_ddict = self._workflow_ddict
        generation_requests = workflow_ddict["generation_requests"]
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            # put prompt in workflow_ddict under request_id
            generation_requests[request_id] = prompt
        workflow_ddict["generation_requests"] = generation_requests

        return prompts

    def train(self, model_config: ModelConfig, workflow_ddict: DDict, use_default_rollout=True, **kwargs):
        from datasets import load_dataset
        self._workflow_ddict = workflow_ddict
        # load model, tokenizer
        model, tokenizer = load_model(model_config)
        local_reward_funcs = []
        flow_reward_funcs = []
        #figure out reward functions, check which are supposed to be used for training or evaluation
        if use_default_rollout:
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                rollout_func=self._default_rollout_func,
                reward_funcs=self._reward_funcs,
                callbacks=self._trainer_callbacks,
                args=self._grpo_config,
                train_dataset=self._dataset,
            )
        else:
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                rollout_func=self._rollout_func,
                reward_funcs=self._reward_funcs,
                callbacks=self._trainer_callbacks,
                args=self._grpo_config,
                train_dataset=self._dataset,
            )

        trainer.train()

