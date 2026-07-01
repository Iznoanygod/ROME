"""Sequential RL flow built on ROSE's ``SequentialReinforcementLearner``.

The flow orchestrates three tiers of async work coordinated through a
shared Dragon ``DDict``:

* **Generators** — long-lived asyncflow tasks that consume prompts from
  ``generator_<i>_input`` and publish ``{prompt_ids, completion_ids,
  logprobs}`` triples into ``generator_<i>_output``. Weights are
  reloaded between batches whenever the trainer bumps
  ``WEIGHT_VERSION_KEY``.
* **Scorers** — one asyncflow task per (``@Workflow.reward_task``,
  scorer_index) pair. Each pulls model outputs from its
  ``reward_<name>_<i>_input`` queue, runs the reward function, writes
  the scalar reward into ``reward_<name>_<i>_output``.
* **Trainer** — the ``rl.update_task``-registered coroutine. Reads
  everything back through :class:`rome.train.GRPO`'s default rollout /
  reward-wrapper helpers.

The scheduler / gatherer coroutines glue the three tiers by moving
records between per-worker queues and the aggregated views the trainer
consumes.
"""

import asyncio
from typing import Any, Callable, List, Optional

import torch
from radical.asyncflow import WorkflowEngine
from rose.learner import SequentialReinforcementLearner
from rose.metrics import GREATER_THAN_THRESHOLD

from dragon.data.ddict import DDict
from dragon.native.event import Event

from rome.config import ModelConfig
from rome.trainer import Trainer
from rome.utils import load_model, read_weight_version, reload_model
from rome.workflow import Workflow


class SequentialFlowConfig:
    """Configuration for :class:`SequentialFlow`.

    Parameters
    ----------
    iterations : int, optional
        Number of outer RL iterations to run. Default 10. ``0`` runs
        until ``reward_threshold`` is met (requires ``reward_threshold``
        to be set).
    reward_threshold : float, optional
        Reward threshold for terminating the flow. Default None
        (iteration cap is the sole stop signal).
    operator : str, optional
        Comparison operator for the stop criterion. Default
        ``GREATER_THAN_THRESHOLD``.
    num_generators : int, optional
        Number of streaming generator tasks. Default 2.
    num_scorers : int, optional
        Number of scorer tasks per ``@Workflow.reward_task`` function.
        Default 2.
    batch_size : int, optional
        Generator batch size (prompts per ``model.generate`` call).
        Default 4.
    """

    def __init__(
        self,
        iterations: Optional[int] = 10,
        reward_threshold: Optional[float] = None,
        operator: Optional[str] = GREATER_THAN_THRESHOLD,
        num_generators: int = 2,
        num_scorers: int = 2,
        batch_size: int = 4,
    ):
        self.iterations = iterations
        self.reward_threshold = reward_threshold
        self.operator = operator
        self.num_generators = num_generators
        self.num_scorers = num_scorers
        self.batch_size = batch_size


class SequentialFlow(Workflow):
    """Iterative RL flow backed by ROSE's ``SequentialReinforcementLearner``.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration for the model and tokenizer.
    trainer : Trainer
        Training algorithm (e.g. :class:`rome.train.GRPO`).
    evaluate_func : Callable
        Per-iteration evaluation function. Called by the stop
        criterion; returns the scalar metric compared against
        ``reward_threshold``.
    asyncflow : WorkflowEngine
        Pre-existing radical.asyncflow engine.
    flow_config : SequentialFlowConfig
        Flow-specific knobs.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        trainer: Trainer,
        evaluate_func: Callable,
        asyncflow: WorkflowEngine,
        flow_config: SequentialFlowConfig,
    ):
        super().__init__(
            model_config=model_config,
            trainer=trainer,
            evaluate_func=evaluate_func,
            asyncflow=asyncflow,
        )
        self.rl = SequentialReinforcementLearner(asyncflow=asyncflow)
        self.flow_config = flow_config
        self._generator_tasks: List[Any] = []
        self._scorer_tasks: List[Any] = []
        self._poll_interval = 0.1  # scheduler / gatherer loop sleep

    # ------------------------------------------------------------------
    # shared-state seeding — factored out so tests can drive it directly
    # ------------------------------------------------------------------
    def _reward_task_funcs(self) -> List[Callable]:
        return [rf for rf in self.trainer.reward_funcs if hasattr(rf, "_is_reward_task")]

    def _seed_state(self, workflow_ddict: Any) -> None:
        """Populate every key the scheduler / gatherer / task coroutines
        will read before they read it. Without this the first read of
        ``generation_requests`` (etc.) KeyErrors on a fresh DDict.
        """
        workflow_ddict["generation_requests"] = {}
        workflow_ddict["generator_outputs"] = {}
        for i in range(self.flow_config.num_generators):
            workflow_ddict[f"generator_{i}_input"] = {}
            workflow_ddict[f"generator_{i}_output"] = {}
        for rf in self._reward_task_funcs():
            workflow_ddict[f"reward_{rf.__name__}_outputs"] = {}
            for i in range(self.flow_config.num_scorers):
                workflow_ddict[f"reward_{rf.__name__}_{i}_input"] = {}
                workflow_ddict[f"reward_{rf.__name__}_{i}_output"] = {}

    # ------------------------------------------------------------------
    # default generator body — model-agnostic HF ``generate`` call
    # ------------------------------------------------------------------
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
            outputs = model.generate(inputs, generation_config=generation_config)
        return outputs

    # ------------------------------------------------------------------
    # scheduler / gatherer coroutines
    # ------------------------------------------------------------------
    async def _generation_schedule(self, workflow_ddict, terminate_event: Event):
        submitted_requests: set = set()
        while not terminate_event.is_set():
            generation_requests = workflow_ddict["generation_requests"]
            new_ids = [
                rid for rid in generation_requests.keys() if rid not in submitted_requests
            ]
            if new_ids:
                generator_queues = [
                    workflow_ddict[f"generator_{i}_input"]
                    for i in range(self.flow_config.num_generators)
                ]
                for rid in new_ids:
                    shortest = min(generator_queues, key=lambda q: len(q))
                    shortest[rid] = generation_requests[rid]
                    submitted_requests.add(rid)
                for i in range(self.flow_config.num_generators):
                    workflow_ddict[f"generator_{i}_input"] = generator_queues[i]
            await asyncio.sleep(self._poll_interval)

    async def _scorer_schedule(self, workflow_ddict, terminate_event: Event):
        submitted_requests: set = set()
        reward_task_funcs = self._reward_task_funcs()
        while not terminate_event.is_set():
            generator_outputs = workflow_ddict["generator_outputs"]
            new_ids = [
                rid for rid in generator_outputs.keys() if rid not in submitted_requests
            ]
            if new_ids and reward_task_funcs:
                scorer_queues = {
                    rf.__name__: [
                        workflow_ddict[f"reward_{rf.__name__}_{i}_input"]
                        for i in range(self.flow_config.num_scorers)
                    ]
                    for rf in reward_task_funcs
                }
                for rid in new_ids:
                    for rf in reward_task_funcs:
                        shortest = min(scorer_queues[rf.__name__], key=lambda q: len(q))
                        shortest[rid] = generator_outputs[rid]
                    submitted_requests.add(rid)
                for i in range(self.flow_config.num_scorers):
                    for rf in reward_task_funcs:
                        workflow_ddict[f"reward_{rf.__name__}_{i}_input"] = scorer_queues[
                            rf.__name__
                        ][i]
            await asyncio.sleep(self._poll_interval)

    async def _generation_gather(self, workflow_ddict, terminate_event: Event):
        while not terminate_event.is_set():
            generator_outputs = workflow_ddict["generator_outputs"]
            for i in range(self.flow_config.num_generators):
                per_worker = workflow_ddict[f"generator_{i}_output"]
                for rid, val in per_worker.items():
                    if rid not in generator_outputs:
                        generator_outputs[rid] = val
            workflow_ddict["generator_outputs"] = generator_outputs
            await asyncio.sleep(self._poll_interval)

    async def _scorer_gather(self, workflow_ddict, terminate_event: Event):
        reward_task_funcs = self._reward_task_funcs()
        while not terminate_event.is_set():
            for rf in reward_task_funcs:
                scorer_outputs = workflow_ddict[f"reward_{rf.__name__}_outputs"]
                for i in range(self.flow_config.num_scorers):
                    per_worker = workflow_ddict[f"reward_{rf.__name__}_{i}_output"]
                    for rid, val in per_worker.items():
                        if rid not in scorer_outputs:
                            scorer_outputs[rid] = val
                workflow_ddict[f"reward_{rf.__name__}_outputs"] = scorer_outputs
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # entrypoint
    # ------------------------------------------------------------------
    async def launch(self, iterations: Optional[int] = None) -> None:
        """Start the sequential RL loop.

        Parameters
        ----------
        iterations : int, optional
            Override ``flow_config.iterations``. ``0`` runs until
            ``reward_threshold`` is met.
        """
        n_iter = iterations if iterations is not None else self.flow_config.iterations
        threshold = self.flow_config.reward_threshold
        if threshold is None and (n_iter is None or n_iter <= 0):
            raise ValueError(
                "SequentialFlow.launch needs either a positive iterations "
                "cap or a reward_threshold to terminate."
            )
        # ROSE's stop criterion requires a threshold; use a sentinel one
        # when only iterations gate termination and break the loop
        # manually on iteration count.
        stop_threshold = threshold if threshold is not None else float("inf")

        workflow_ddict = DDict()
        terminate_event = Event()
        self._seed_state(workflow_ddict)

        asyncflow = self.asyncflow
        rl = self.rl
        trainer = self.trainer
        model_config = self.model_config
        evaluate_func = self.evaluate_func

        @asyncflow.function_task
        async def generation_task(
            model_config,
            batch_size,
            _terminate_event,
            _workflow_ddict,
            _input_key,
            _output_key,
        ):
            processed: set = set()
            model, tokenizer = load_model(model_config)
            current_weight_version = read_weight_version(_workflow_ddict)

            while not _terminate_event.is_set():
                requests = _workflow_ddict[_input_key]
                pending = [rid for rid in requests.keys() if rid not in processed]
                if pending:
                    for i in range(0, len(pending), batch_size):
                        new_version = read_weight_version(
                            _workflow_ddict, default=current_weight_version
                        )
                        if new_version != current_weight_version:
                            reload_model(model, model_config)
                            current_weight_version = new_version

                        batch = pending[i : i + batch_size]
                        prompts = [requests[rid] for rid in batch]
                        outputs = SequentialFlow._default_generator_func(
                            prompts, model, tokenizer, model_config.generation_config
                        )
                        transition_scores = model.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                        prompt_ids = tokenizer.apply_chat_template(
                            prompts,
                            add_generation_prompt=True,
                            tokenize=True,
                            padding=False,
                            return_tensors=None,
                        )
                        out_dict = _workflow_ddict[_output_key]
                        for j, rid in enumerate(batch):
                            out_dict[rid] = {
                                "prompt_ids": prompt_ids[j],
                                "completion_ids": outputs.sequences[j],
                                "logprobs": transition_scores[j],
                            }
                            processed.add(rid)
                        _workflow_ddict[_output_key] = out_dict
                await asyncio.sleep(0.05)

        @asyncflow.function_task
        async def scorer_task(
            reward_func, _terminate_event, _workflow_ddict, _input_key, _output_key
        ):
            scored: set = set()
            while not _terminate_event.is_set():
                inputs = _workflow_ddict[_input_key]
                pending = [rid for rid in inputs.keys() if rid not in scored]
                if pending:
                    out_dict = _workflow_ddict[_output_key]
                    for rid in pending:
                        score = reward_func(inputs[rid])
                        out_dict[rid] = score
                        scored.add(rid)
                    _workflow_ddict[_output_key] = out_dict
                await asyncio.sleep(0.05)

        @rl.update_task(as_executable=False)
        async def train_model(model_config=model_config, workflow_ddict=workflow_ddict):
            return await trainer.train(model_config, workflow_ddict=workflow_ddict)

        @rl.as_stop_criterion(
            metric_name="MODEL_REWARD",
            threshold=stop_threshold,
            operator=self.flow_config.operator,
            as_executable=False,
        )
        async def test_model():
            return await evaluate_func(model_config)

        # start generators
        for i in range(self.flow_config.num_generators):
            self._generator_tasks.append(
                generation_task(
                    model_config=self.model_config,
                    batch_size=self.flow_config.batch_size,
                    _terminate_event=terminate_event,
                    _workflow_ddict=workflow_ddict,
                    _input_key=f"generator_{i}_input",
                    _output_key=f"generator_{i}_output",
                )
            )
        # start scorers (only for @reward_task-marked funcs)
        for i in range(self.flow_config.num_scorers):
            for rf in self._reward_task_funcs():
                self._scorer_tasks.append(
                    scorer_task(
                        reward_func=rf,
                        _workflow_ddict=workflow_ddict,
                        _terminate_event=terminate_event,
                        _input_key=f"reward_{rf.__name__}_{i}_input",
                        _output_key=f"reward_{rf.__name__}_{i}_output",
                    )
                )

        # background scheduler / gatherer coroutines
        schedule_gather_fut = asyncio.gather(
            self._generation_schedule(workflow_ddict, terminate_event),
            self._generation_gather(workflow_ddict, terminate_event),
            self._scorer_schedule(workflow_ddict, terminate_event),
            self._scorer_gather(workflow_ddict, terminate_event),
            return_exceptions=True,
        )

        try:
            async for state in rl.start():
                print(f"Iteration {state.iteration}: metric={state.metric_value}")
                if n_iter and n_iter > 0 and state.iteration >= n_iter:
                    break
        finally:
            terminate_event.set()
            try:
                await schedule_gather_fut
            except asyncio.CancelledError:
                pass
