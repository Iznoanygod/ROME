import os
import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
from rome.trainer import Trainer

class GRPO(Trainer):
    def __init__(
        self,
        required_gpus: int = 1,
        training_kwargs: Optional[Dict[str, Any]] = None,
        rollout_func: Optional[Callable] = None,
        reward_funcs: Optional[List[Callable]] = None,
        on_step_end: Optional[Callable] = None,
        prompt_gen_batch_size: int = 2,
    ):
        super().__init__(required_gpus=required_gpus, training_kwargs=training_kwargs, reward_funcs=reward_funcs)
        self.rollout_func = rollout_func
        self.on_step_end = on_step_end
        self.prompt_gen_batch_size = prompt_gen_batch_size
        self.rollout_func = rollout_func or None
        self.reward_funcs = reward_funcs or []
        self.on_step_end = on_step_end
        self.prompt_gen_batch_size = prompt_gen_batch_size
        self.output_dir = output_dir
        self.run_name = run_name
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def _make_grpo_config(self):
        # import lazily to avoid top-level dependency unless used
        from trl import GRPOConfig
        default = dict(
            learning_rate=5e-6,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=self.prompt_gen_batch_size * 4,
            gradient_accumulation_steps=4,
            num_generations=self.prompt_gen_batch_size,
            generation_batch_size=self.prompt_gen_batch_size * 4,
            save_strategy="no",
            max_completion_length=1024,
            max_steps=1000,
            save_steps=100,
            max_grad_norm=1.0,
            report_to="none",
            run_name=self.run_name,
            output_dir=self.output_dir,
            overwrite_output_dir=True,
        )
        default.update(self.training_kwargs)
        return GRPOConfig(**default)
    
    def _make_callback_class(self, reset_event):
        from transformers import TrainerCallback
        import shutil, os

        outer_on_step_end = self.on_step_end

        class _GRPOCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                # call user-supplied handler if provided (keeps original behaviour pluggable)
                pass

        return _GRPOCallback

    def run_training(self, reset_event, dataset, prompt_output_ddict, generated_ddict):
        """
        Blocking call that loads model/tokenizer, constructs GRPOTrainer and runs trainer.train()
        Intended to be called inside the flow.function_task wrapper.
        """
        # lazy imports
        from datasets import Dataset
        from trl import GRPOTrainer
        model, tokenizer = self._load_model_and_tokenizer()
        grpo_config = self._make_grpo_config()

        # default rollout_func and reward_funcs must be provided by user or provided at wrapper init
        #if self.rollout_func is None:
        #    raise RuntimeError("rollout_func must be provided to GRPOTrainerWrapper")
        if not self.reward_funcs:
            raise RuntimeError("reward_funcs must be provided to GRPOTrainerWrapper")

        # build dataset from keys (simple formatter consistent with existing code)

        CallbackClass = self._make_callback_class(reset_event, superfamily_ddict, gen_fam_ddict, folded_ddict, scored_ddict)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            rollout_func=self.rollout_func,
            reward_funcs=self.reward_funcs,
            args=grpo_config,
            train_dataset=dataset,
            callbacks=[CallbackClass()],
        )
        trainer.train()
        return "done"