import os
import logging
import pathlib import Path

from rome.trainer import Trainer

class SFT(Trainer):
    def __init__(
        self,
        *,
        gpus: int = 1,
        dataset,
        reward_funcs: List[Callable],
        trainer_callbacks: Optional[List[Any]] = None,
        sft_config: Optional[SFTConfig] = None,
        top_p: float = 0.5,
    ):
        super().__init__(gpus=gpus, dataset=dataset, reward_funcs=reward_funcs)
        self._trainer_callbacks = trainer_callbacks
        self._sft_config = sft_config
        if self._sft_config is None:
            self._sft_config = SFTConfig(
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

                num_train_epochs=3,
            )
        self._top_p = top_p

