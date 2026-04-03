import os
import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

class GRPO:
    def __init__(
        self,
        model_path: str = "GreatCaptainNemo/ProLLaMA",
        lora_id: Optional[str] = "prolora",
        device_map: Any = "auto",
        training_kwargs: Optional[Dict[str, Any]] = None,
        rollout_func: Optional[Callable] = None,
        reward_funcs: Optional[List[Callable]] = None,
        on_step_end: Optional[Callable] = None,
        prompt_gen_batch_size: int = 2,
        output_dir: str = "prolora",
        run_name: str = "prolora-rome",
    ):
        self.model_path = model_path
        self.lora_id = lora_id
        self.device_map = device_map
        self.training_kwargs = training_kwargs or {}
        self.rollout_func = rollout_func
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
    
    def _make_callback_class(self, reset_event, superfamily_ddict=None, gen_fam_ddict=None, folded_ddict=None, scored_ddict=None):
        from transformers import TrainerCallback
        import shutil, os

        outer_on_step_end = self.on_step_end

        class _GRPOCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                # call user-supplied handler if provided (keeps original behaviour pluggable)
                if outer_on_step_end:
                    try:
                        outer_on_step_end(args, state, control, superfamily_ddict, gen_fam_ddict, folded_ddict, scored_ddict, reset_event)
                    except Exception as e:
                        # swallow to avoid breaking trainer loop
                        logging.getLogger("GRPOTrainerWrapper").exception("on_step_end handler failed: %s", e)
                # default: set reset_event so external launchers can react
                try:
                    reset_event.set()
                except Exception:
                    pass

        return _GRPOCallback

    def run_training(self, reset_event, superfamily_ddict=None, gen_fam_ddict=None, folded_ddict=None, scored_ddict=None):
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
        if self.rollout_func is None:
            raise RuntimeError("rollout_func must be provided to GRPOTrainerWrapper")
        if not self.reward_funcs:
            raise RuntimeError("reward_funcs must be provided to GRPOTrainerWrapper")

        # build dataset from keys (simple formatter consistent with existing code)
        formatted_data = [{"prompt": s} for s in (superfamily_ddict.keys() if superfamily_ddict is not None else [])]
        train_dataset = Dataset.from_list(formatted_data) if formatted_data else Dataset.from_list([])

        CallbackClass = self._make_callback_class(reset_event, superfamily_ddict, gen_fam_ddict, folded_ddict, scored_ddict)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            rollout_func=self.rollout_func,
            reward_funcs=self.reward_funcs,
            args=grpo_config,
            train_dataset=train_dataset,
            callbacks=[CallbackClass()],
        )
        trainer.train()
        return "done"