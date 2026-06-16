import asyncio
from typing import Callable, List, Optional

import torch
from radical.asyncflow import WorkflowEngine
from rose.learner import SequentialReinforcementLearner
from rose.metrics import GREATER_THAN_THRESHOLD

from dragon.data.ddict import DDict
from dragon.native.event import Event

from rome.config import ModelConfig
from rome.trainer import Trainer
from rome.utils import (
    WEIGHT_PATH_KEY,
    WEIGHT_VERSION_KEY,
    WeightSyncCallback,
    load_model,
    maybe_reload_weights,
)
from rome.workflow import Workflow


class StreamFlowConfig:
    """Configuration for StreamFlow.
    Parameters
    ----------
    iterations : int, optional
        Number of trainer iterations. Default is 10.
    reward_threshold : float, optional
        Reward threshold for terminating the flow. Default is None.
    operator : str, optional
        Comparison operator for the stop criterion. Default is
        ``GREATER_THAN_THRESHOLD``.
    num_generators : int, optional
        Number of generator stream tasks running concurrently. Default 2.
    num_scorers : int, optional
        Number of scorer stream tasks per reward function. Default 2.
    batch_size : int, optional
        Generator batch size (prompts per ``model.generate`` call). Default 4.
    num_generations_per_prompt : int, optional
        Number of completions the rollout will pull per prompt. Default 4.
    prompts : list[str], optional
        Static prompt pool the generators continuously stream completions
        for. If ``None``, the rollout must populate ``generation_prompts``
        in the workflow ddict.
    max_buffer_per_prompt : int, optional
        Soft upper bound on completions buffered per prompt before
        generators throttle. Default is 32.
    checkpoint_dir : str, optional
        Directory where the trainer writes versioned weight checkpoints
        (``{checkpoint_dir}/step_{version}``) for streaming generators to
        pick up. If ``None``, weight syncing is disabled and generators
        keep the model they loaded at startup.
    checkpoint_interval : int, optional
        How many trainer steps between checkpoint writes. Default 1.
    """
    def __init__(
        self,
        iterations: Optional[int] = 10,
        reward_threshold: Optional[float] = None,
        operator: Optional[str] = GREATER_THAN_THRESHOLD,
        num_generators: int = 2,
        num_scorers: int = 2,
        batch_size: int = 4,
        num_generations_per_prompt: int = 4,
        prompts: Optional[List[str]] = None,
        max_buffer_per_prompt: int = 32,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1,
    ):
        self.iterations = iterations
        self.reward_threshold = reward_threshold
        self.operator = operator
        self.num_generators = num_generators
        self.num_scorers = num_scorers
        self.batch_size = batch_size
        self.num_generations_per_prompt = num_generations_per_prompt
        self.prompts = list(prompts) if prompts is not None else []
        self.max_buffer_per_prompt = max_buffer_per_prompt
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval


class StreamFlow(Workflow):
    """Streaming RL flow.
    Unlike :class:`SequentialFlow`, generators do not wait for explicit
    request ids from the trainer. They continuously consume prompts from a
    shared pool (``generation_prompts``) and append completions to a
    per-prompt buffer (``generator_outputs[prompt]`` -> list). Scorers do
    the same on the completion side, producing scores keyed by the index
    of the completion in the per-prompt buffer.
    The trainer's rollout pulls ``num_generations_per_prompt`` completions
    per prompt out of the buffer, draining them as they are consumed so
    later steps see fresh generations.
    Parameters
    ----------
    model_config : ModelConfig
        Model configuration for the model and tokenizer.
    trainer : Trainer
        Training algorithm (e.g. ``GRPO``).
    evaluate_func : Callable
        Per-iteration evaluation function. Same conventions as
        :class:`SequentialFlow`.
    asyncflow : WorkflowEngine
        radical.asyncflow engine for task placement.
    flow_config : StreamFlowConfig
        Stream-specific knobs.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        trainer: Trainer,
        evaluate_func: Callable,
        asyncflow: WorkflowEngine,
        flow_config: StreamFlowConfig,
    ):
        super().__init__(
            model_config=model_config,
            trainer=trainer,
            evaluate_func=evaluate_func,
            asyncflow=asyncflow,
        )
        self.rl = SequentialReinforcementLearner(asyncflow=asyncflow)
        self.flow_config = flow_config
        self._generator_tasks = []
        self._scorer_tasks = []

    @staticmethod
    def _default_generator_func(prompts, model, tokenizer, generation_config):
        inputs = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                generation_config=generation_config,
            )
        return outputs

    async def _prompt_schedule(self, workflow_ddict, terminate_event: Event):
        """Balance the prompt pool across per-generator input queues.
        Generators consume from ``generator_{i}_input`` which maps
        ``prompt -> prompt`` (the value is unused; the key is what gets
        generated for). Each generator sees roughly the same set of
        prompts and produces completions for each one in a loop.
        """
        while not terminate_event.is_set():
            generation_prompts = workflow_ddict["generation_prompts"]
            generator_queues = [
                workflow_ddict[f"generator_{i}_input"]
                for i in range(self.flow_config.num_generators)
            ]
            # mirror the prompt pool into every generator queue
            for prompt in generation_prompts:
                for q in generator_queues:
                    if prompt not in q:
                        q[prompt] = prompt
            for i in range(self.flow_config.num_generators):
                workflow_ddict[f"generator_{i}_input"] = generator_queues[i]

    async def _generation_gather(self, workflow_ddict, terminate_event: Event):
        """Append new per-generator completions into ``generator_outputs``.
        ``generator_outputs[prompt]`` is a list of generation dicts. New
        items from each generator are appended; the rollout consumes from
        the head of the list.
        """
        seen = {}
        while not terminate_event.is_set():
            generator_outputs = workflow_ddict["generator_outputs"]
            for i in range(self.flow_config.num_generators):
                produced = workflow_ddict[f"generator_{i}_output"]
                for prompt, completions in produced.items():
                    promoted = seen.get((i, prompt), 0)
                    if promoted >= len(completions):
                        continue
                    bucket = generator_outputs.get(prompt, [])
                    bucket.extend(completions[promoted:])
                    generator_outputs[prompt] = bucket
                    seen[(i, prompt)] = len(completions)
            workflow_ddict["generator_outputs"] = generator_outputs

    async def _scorer_schedule(self, workflow_ddict, terminate_event: Event):
        """Fan generator outputs out to scorer input queues per reward func.
        Each scorer queue ``reward_{name}_{i}_input`` is keyed by
        ``(prompt, completion_index)`` so partial scoring across a buffer
        is well defined.
        """
        submitted = set()
        while not terminate_event.is_set():
            generator_outputs = workflow_ddict["generator_outputs"]

            scorer_queues = {
                rf.__name__: [
                    workflow_ddict[f"reward_{rf.__name__}_{i}_input"]
                    for i in range(self.flow_config.num_scorers)
                ]
                for rf in self.trainer.reward_funcs
            }

            for prompt, completions in generator_outputs.items():
                for idx, completion in enumerate(completions):
                    key = (prompt, idx)
                    if key in submitted:
                        continue
                    for rf in self.trainer.reward_funcs:
                        queues = scorer_queues[rf.__name__]
                        shortest = min(queues, key=lambda q: len(q))
                        shortest[key] = completion
                    submitted.add(key)

            for rf in self.trainer.reward_funcs:
                for i in range(self.flow_config.num_scorers):
                    workflow_ddict[f"reward_{rf.__name__}_{i}_input"] = (
                        scorer_queues[rf.__name__][i]
                    )

    async def _scorer_gather(self, workflow_ddict, terminate_event: Event):
        """Merge per-scorer outputs into ``reward_{name}_outputs``."""
        while not terminate_event.is_set():
            for rf in self.trainer.reward_funcs:
                merged = workflow_ddict[f"reward_{rf.__name__}_outputs"]
                for i in range(self.flow_config.num_scorers):
                    produced = workflow_ddict[f"reward_{rf.__name__}_{i}_output"]
                    for key, score in produced.items():
                        if key not in merged:
                            merged[key] = score
                workflow_ddict[f"reward_{rf.__name__}_outputs"] = merged

    async def launch(self, iterations: Optional[int] = None) -> None:
        """Start the streaming RL loop."""
        workflow_ddict = DDict()
        terminate_event = Event()
        asyncflow = self.asyncflow
        rl = self.rl
        trainer = self.trainer
        model_config = self.model_config
        evaluate_func = self.evaluate_func

        # seed the prompt pool and weight-sync state
        workflow_ddict["generation_prompts"] = list(self.flow_config.prompts)
        workflow_ddict["generator_outputs"] = {}
        workflow_ddict[WEIGHT_VERSION_KEY] = 0
        workflow_ddict[WEIGHT_PATH_KEY] = None
        for i in range(self.flow_config.num_generators):
            workflow_ddict[f"generator_{i}_input"] = {}
            workflow_ddict[f"generator_{i}_output"] = {}
        for rf in self.trainer.reward_funcs:
            workflow_ddict[f"reward_{rf.__name__}_outputs"] = {}
            for i in range(self.flow_config.num_scorers):
                workflow_ddict[f"reward_{rf.__name__}_{i}_input"] = {}
                workflow_ddict[f"reward_{rf.__name__}_{i}_output"] = {}

        max_buffer = self.flow_config.max_buffer_per_prompt

        @asyncflow.function_task
        async def generation_task(
            model_config,
            batch_size,
            _terminate_event,
            _workflow_ddict,
            _input_key,
            _output_key,
            _max_buffer,
        ):
            model, tokenizer = load_model(model_config)
            local_version = 0

            while not _terminate_event.is_set():
                prompts_to_run = []
                input_dict = _workflow_ddict[_input_key]
                output_dict = _workflow_ddict[_output_key]
                for prompt in list(input_dict.keys()):
                    existing = output_dict.get(prompt, [])
                    if len(existing) >= _max_buffer:
                        continue
                    prompts_to_run.append(prompt)

                if not prompts_to_run:
                    await asyncio.sleep(0.1)
                    continue

                for i in range(0, len(prompts_to_run), batch_size):
                    # Between batches, swap in a fresh adapter if the
                    # trainer has published one.
                    model, local_version = maybe_reload_weights(
                        model, model_config, _workflow_ddict, local_version,
                    )
                    batch = prompts_to_run[i : i + batch_size]
                    outputs = StreamFlow._default_generator_func(
                        batch, model, tokenizer, model_config.generation_config
                    )
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences,
                        outputs.scores,
                        normalize_logits=True,
                    )
                    prompt_ids = tokenizer.apply_chat_template(
                        batch,
                        add_generation_prompt=True,
                        tokenize=True,
                        padding=False,
                        return_tensors=None,
                    )
                    for j, prompt in enumerate(batch):
                        bucket = output_dict.get(prompt, [])
                        bucket.append({
                            "prompt_ids": prompt_ids[j],
                            "completion_ids": outputs.sequences[j],
                            "logprobs": transition_scores[j],
                        })
                        output_dict[prompt] = bucket
                    _workflow_ddict[_output_key] = output_dict

        @asyncflow.function_task
        async def scorer_task(
            reward_func,
            _terminate_event,
            _workflow_ddict,
            _input_key,
            _output_key,
        ):
            scored_keys = set()
            while not _terminate_event.is_set():
                inputs = _workflow_ddict[_input_key]
                output_dict = _workflow_ddict[_output_key]
                pending = [k for k in inputs.keys() if k not in scored_keys]
                if not pending:
                    await asyncio.sleep(0.1)
                    continue
                for key in pending:
                    completion = inputs[key]
                    score = reward_func(completion)
                    output_dict[key] = score
                    scored_keys.add(key)
                _workflow_ddict[_output_key] = output_dict

        # Inject the weight-sync callback so the trainer writes a fresh
        # checkpoint (and bumps the version key) at every step.
        if self.flow_config.checkpoint_dir is not None:
            weight_sync_cb = WeightSyncCallback(
                workflow_ddict=workflow_ddict,
                model_config=model_config,
                checkpoint_dir=self.flow_config.checkpoint_dir,
                interval=self.flow_config.checkpoint_interval,
            )
            existing_cbs = getattr(trainer, "_trainer_callbacks", None) or []
            trainer._trainer_callbacks = list(existing_cbs) + [weight_sync_cb]

        @rl.update_task(as_executable=False)
        async def train_model(model_config=model_config, workflow_ddict=workflow_ddict):
            return await trainer.train(model_config, workflow_ddict=workflow_ddict)

        @rl.as_stop_criterion(
            metric_name='MODEL_REWARD',
            threshold=self.flow_config.reward_threshold or 128,
            operator=self.flow_config.operator,
            as_executable=False,
        )
        async def test_model():
            return await evaluate_func(model_config)

        # spawn generators
        for i in range(self.flow_config.num_generators):
            self._generator_tasks.append(generation_task(
                model_config=self.model_config,
                batch_size=self.flow_config.batch_size,
                _terminate_event=terminate_event,
                _workflow_ddict=workflow_ddict,
                _input_key=f"generator_{i}_input",
                _output_key=f"generator_{i}_output",
                _max_buffer=max_buffer,
            ))

        # spawn scorers — only those marked as reward tasks run inside the flow
        reward_task_funcs = [
            rf for rf in self.trainer.reward_funcs
            if hasattr(rf, "_is_reward_task")
        ]
        for i in range(self.flow_config.num_scorers):
            for rf in reward_task_funcs:
                self._scorer_tasks.append(scorer_task(
                    reward_func=rf,
                    _terminate_event=terminate_event,
                    _workflow_ddict=workflow_ddict,
                    _input_key=f"reward_{rf.__name__}_{i}_input",
                    _output_key=f"reward_{rf.__name__}_{i}_output",
                ))

        schedule_gather_fut = asyncio.gather(
            self._prompt_schedule(workflow_ddict, terminate_event),
            self._generation_gather(workflow_ddict, terminate_event),
            self._scorer_schedule(workflow_ddict, terminate_event),
            self._scorer_gather(workflow_ddict, terminate_event),
        )

        async for state in rl.start():
            print(f"Iteration {state.iteration}: metric={state.metric_value}")

        terminate_event.set()
        await schedule_gather_fut
        return
