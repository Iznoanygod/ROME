import asyncio
import time
from typing import Any, Callable, List, Optional

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from rome.config import ModelConfig
from rome.trainer import Trainer
from rome.utils import bump_weight_version, load_model, save_model

from dragon.data.ddict import DDict


class GRPO(Trainer):
    """GRPO trainer.

    See ``trl.GRPOTrainer`` for the underlying implementation of the GRPO
    algorithm, and ``trl.GRPOConfig`` for training knobs.

    Parameters
    ----------
    gpus : int
        Number of GPUs to use for training.
    reward_funcs : List[Callable]
        List of reward functions to use for training. Reward functions
        run inside the trainer unless marked with the
        ``@Workflow.reward_task`` decorator, in which case the workflow
        is responsible for computing them as separate tasks and the
        trainer reads their results from the shared ``workflow_ddict``.
    dataset : Dataset, optional
        Training dataset. Pass this in at construction so
        :class:`SequentialFlow` can call ``await trainer.train(...)``
        without needing to plumb the dataset through its update task.
    trainer_callbacks : Optional[List[Any]]
        List of callbacks for the trainer.
    rollout_func : Optional[Callable]
        Rollout function for generating trajectories. If not provided,
        the built-in workflow-aware rollout is used (reads / writes the
        shared ``workflow_ddict`` used by the flow's generator tasks).
    grpo_config : Optional[GRPOConfig]
        Configuration for the GRPO trainer. If not provided, ROME
        defaults are used.
    """

    def __init__(
        self,
        *,
        gpus: int = 1,
        reward_funcs: List[Callable],
        dataset: Optional[Dataset] = None,
        trainer_callbacks: Optional[List[Any]] = None,
        rollout_func: Optional[Callable] = None,
        grpo_config: Optional[GRPOConfig] = None,
    ):
        super().__init__(gpus=gpus, reward_funcs=reward_funcs, dataset=dataset)
        self._trainer_callbacks = trainer_callbacks
        self._rollout_func = rollout_func
        self._grpo_config = grpo_config
        if self._grpo_config is None:
            self._grpo_config = GRPOConfig(
                # Parameters that control training
                learning_rate=5e-6,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                optim="adamw_8bit",
                logging_steps=1,
                num_train_epochs=3,
                # generation
                per_device_train_batch_size=4,
                gradient_accumulation_steps=16,
                num_generations=4,
                generation_batch_size=4,
            )

    def _reward_func_wrapper(self, reward_func) -> Callable:
        """Wrap a ``@Workflow.reward_task``-marked reward func.

        Instead of computing the reward inline, the wrapper polls
        ``workflow_ddict[reward_<name>_outputs]`` for scores that the
        flow's scorer tasks have already produced.
        """
        async def _wrapped_reward_func(prompts, completions, ground_truths, **kwargs):
            request_ids = kwargs["request_ids"]
            workflow_ddict = self._workflow_ddict
            identifier = reward_func.__name__
            reward_outputs = workflow_ddict[f"reward_{identifier}_outputs"]
            while not all(request_id in reward_outputs for request_id in request_ids):
                await asyncio.sleep(1)
                reward_outputs = workflow_ddict[f"reward_{identifier}_outputs"]
            return [reward_outputs[request_id] for request_id in request_ids]

        return _wrapped_reward_func

    def _default_rollout_func(self, prompts: list, trainer: GRPOTrainer, **kwargs):
        """Workflow-aware default rollout.

        Publishes each prompt into ``generation_requests`` under a fresh
        UUID and blocks until the flow's generator tasks have produced a
        matching entry in ``generator_outputs``. Returns the batched
        prompt/completion/logprob triples in the order the trainer asked
        for.
        """
        import uuid

        workflow_ddict = self._workflow_ddict
        generation_requests = workflow_ddict["generation_requests"]
        request_ids = []
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            generation_requests[request_id] = prompt
            request_ids.append(request_id)
        workflow_ddict["generation_requests"] = generation_requests

        prompt_ids, completion_ids, logprobs_ids = [], [], []
        generator_outputs = workflow_ddict["generator_outputs"]
        for request_id in request_ids:
            while request_id not in generator_outputs:
                time.sleep(1)
                generator_outputs = workflow_ddict["generator_outputs"]
            model_output = generator_outputs[request_id]
            prompt_ids.append(model_output["prompt_ids"])
            completion_ids.append(model_output["completion_ids"])
            logprobs_ids.append(model_output["logprobs"])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs_ids,
            "request_ids": request_ids,
        }

    async def train(
        self,
        model_config: ModelConfig,
        workflow_ddict: Optional[DDict] = None,
        use_default_rollout: bool = True,
        **kwargs,
    ) -> Any:
        """Fit the model. Dataset comes from the constructor.

        Uses the workflow's shared ``workflow_ddict`` when running under
        :class:`SequentialFlow` so the rollout / task-reward wrappers can
        route via the flow's generator + scorer tasks.
        """
        if self._dataset is None:
            raise ValueError(
                "GRPO requires a dataset — pass one to the constructor "
                "before calling train()."
            )

        self._workflow_ddict = workflow_ddict

        # Split reward functions: local ones run inside trl; task ones
        # (decorated with @Workflow.reward_task) get replaced with the
        # ddict-polling wrapper so their scores come from the flow's
        # scorer tasks.
        local_reward_funcs, flow_reward_funcs = [], []
        for reward_func in self._reward_funcs:
            if hasattr(reward_func, "_is_reward_task"):
                flow_reward_funcs.append(reward_func)
            else:
                local_reward_funcs.append(reward_func)
        wrapped_reward_funcs = local_reward_funcs + [
            self._reward_func_wrapper(rf) for rf in flow_reward_funcs
        ]

        model, tokenizer = load_model(model_config)

        rollout_func = (
            self._default_rollout_func if use_default_rollout else self._rollout_func
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            rollout_func=rollout_func,
            reward_funcs=wrapped_reward_funcs,
            callbacks=self._trainer_callbacks,
            args=self._grpo_config,
            train_dataset=self._dataset,
        )

        # trl's GRPOTrainer.train() is synchronous; offload so the flow's
        # event loop (scheduler / gather / other tasks) keeps making
        # progress while training runs.
        await asyncio.to_thread(trainer.train)
        save_model(model, model_config)
        if workflow_ddict is not None:
            bump_weight_version(workflow_ddict)
        return trainer
