import asyncio
from typing import Any, Callable, Optional

import torch
from radical.asyncflow import WorkflowEngine
from rose.learner import SequentialReinforcementLearner
from rose.metrics import GREATER_THAN_THRESHOLD
from transformers import GenerationConfig

from dragon.data.ddict import DDict
from dragon.native.event import Event

from rome.config import ModelConfig
from rome.trainer import Trainer
from rome.utils import load_model
from rome.workflow import Workflow

class SequentialFlowConfig():
    """Configuration for SequentialFlow.

    Parameters
    ----------
    iterations : int, optional
        Number of iterations to run the flow. Default is 10.
    reward_threshold : float, optional
        Reward threshold for terminating the flow. Default is None.
    operator : str, optional
        Operator to use for comparing rewards. Default is GREATER_THAN_THRESHOLD.
    num_generators : int, optional
        Number of generator tasks. Default is 2.
    num_scorers : int, optional
        Number of scorer tasks for each reward function. Default is 2.
    batch_size : int, optional
        Generator batch size. Default is 4.
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
    """Single iterative RL flow backed by ROSE's SequentialReinforcementLearner.
    
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
    asyncflow : WorkflowEngine, optional
        Pre-existing radical.asyncflow engine.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        trainer: Trainer,
        evaluate_func: Callable,
        asyncflow: WorkflowEngine,
        flow_config:SequentialFlowConfig,
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

    def _default_generator_func(prompts: list[str], model, tokenizer, generation_config):
        inputs = tokenizer.apply_chat_template(
            prompts, 
            add_generation_prompt=True, 
            tokenize=True, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                generation_config=generation_config,
            )
            return outputs
        
    async def _generation_schedule(self, workflow_ddict, terminate_event: Event):
        submitted_requests = []
        while not terminate_event.is_set():
            requests_to_submit = []
            generation_requests = workflow_ddict["generation_requests"]
            for request_id in generation_requests.keys():
                if request_id not in submitted_requests:
                    requests_to_submit.append(request_id)
            # balance requests between generators
            generator_queues = [workflow_ddict[f"generator_{i}_input"] for i in range(self.flow_config.num_generators)]
            for request_id in requests_to_submit:
                # find generator with shortest queue
                shortest_queue = min(generator_queues, key=lambda q: len(q))
                shortest_queue[request_id] = generation_requests[request_id]
                submitted_requests.append(request_id)
            # update generator queues in workflow_ddict
            for i in range(self.flow_config.num_generators):
                workflow_ddict[f"generator_{i}_input"] = generator_queues[i]

    async def _scorer_schedule(self, workflow_ddict, terminate_event: Event):
        submitted_requests = []
        while not terminate_event.is_set():
            requests_to_submit = []
            generator_outputs = workflow_ddict["generator_outputs"]
            for request_id in generator_outputs.keys():
                if request_id not in submitted_requests:
                    requests_to_submit.append(request_id)
            
            scorer_queues = {}
            for reward_func in self.trainer.reward_funcs:
                scorer_queues[reward_func.__name__] = [workflow_ddict[f"reward_{reward_func.__name__}_{i}_input"] for i in range(self.flow_config.num_scorers)]

            for request_id in requests_to_submit:
                for reward_func in self.trainer.reward_funcs:
                    # find scorer with shortest queue for this reward func
                    shortest_queue = min(scorer_queues[reward_func.__name__], key=lambda q: len(q))
                    shortest_queue[request_id] = generator_outputs[request_id]
                submitted_requests.append(request_id)
            for i in range(self.flow_config.num_scorers):
                for reward_func in self.trainer.reward_funcs:
                    workflow_ddict[f"reward_{reward_func.__name__}_{i}_input"] = scorer_queues[reward_func.__name__][i]

    async def _generation_gather(self, workflow_ddict, terminate_event: Event):
        while not terminate_event.is_set():
            generator_outputs = workflow_ddict["generator_outputs"]

            for i in range(self.flow_config.num_generators):
                # check if generator i has requests to process in workflow_ddict, if so, schedule generation task for those requests
                output_key = f"generator_{i}_output"
                output_dict = workflow_ddict[output_key]
                for request_id in output_dict.keys():
                    if request_id not in generator_outputs:
                        generator_outputs[request_id] = output_dict[request_id]
                    
            workflow_ddict["generator_outputs"] = generator_outputs

    async def _scorer_gather(self, workflow_ddict, terminate_event: Event):
        while not terminate_event.is_set():
            for reward_func in self.trainer.reward_funcs:
                scorer_outputs = workflow_ddict[f"reward_{reward_func.__name__}_outputs"]
                for i in range(self.flow_config.num_scorers):
                    output_key = f"reward_{reward_func.__name__}_{i}_output"
                    output_dict = workflow_ddict[output_key]
                    for request_id in output_dict.keys():
                        if request_id not in scorer_outputs:
                            scorer_outputs[request_id] = output_dict[request_id]
                    
                workflow_ddict[f"reward_{reward_func.__name__}_outputs"] = scorer_outputs


    async def launch(self, iterations: Optional[int] = None) -> None:
        """Start the sequential RL loop.

        Parameters
        ----------
        iterations : int, optional
            Override ``flow_config.iterations``. ``0`` runs until the reward
            threshold is met (requires ``reward_threshold`` to be set).
        """

        # create shared dictionary for workflows
        workflow_ddict = DDict()
        terminate_event = Event()
        asyncflow = self.asyncflow
        rl = self.rl
        trainer = self.trainer
        model_config = self.model_config
        evaluate_func = self.evaluate_func

        @asyncflow.function_task
        async def generation_task(model_config, batch_size, _terminate_event, _workflow_ddict, _input_key, _output_key):
            generated_requests = []

            # load models
            model, tokenizer = load_model(model_config)
            #generation config
            
            while not _terminate_event.is_set():
                requests_to_process = []
                # requests is dictionary request_id -> prompt
                requests = _workflow_ddict[_input_key]
                for request_id in requests.keys():
                    if request_id in generated_requests:
                        continue
                    # add to processing list
                    requests_to_process.append(request_id)
                
                # process requests
                if len(requests_to_process) > 0:
                    for i in range(0, len(requests_to_process), batch_size):
                        batch = requests_to_process[i:i+batch_size]
                        # generate outputs for batch
                        prompts = [requests[request_id] for request_id in batch]
                        outputs = _default_generator_func(prompts, model, tokenizer, model_config.generation_config)

                        # put outputs in workflow_ddict
                        output_dict = _workflow_ddict[_output_key]
                        # need to extract from model outputs and put in output dict
                        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                        sequences = outputs.sequences.cpu().numpy()
                        prompt_ids = tokenizer.apply_chat_template(
                            prompts, 
                            add_generation_prompt=True, 
                            tokenize=True, 
                            padding=False, 
                            return_tensors=None
                        )
                        for j, request_id in enumerate(batch):
                            output_dict[request_id] = {
                                "prompt_ids": prompt_ids[j],
                                "completion_ids": outputs.sequences[j],
                                "logprobs": transition_scores[j],
                            }
                            
                        _workflow_ddict[_output_key] = output_dict
                        
        @asyncflow.function_task
        async def scorer_task(reward_func, _terminate_event, _workflow_ddict, _input_key, _output_key):
            scored_requests = []

            while not _terminate_event.is_set():
                requests_to_score = []
                generator_outputs = _workflow_ddict[_input_key]
                for request_id in generator_outputs.keys():
                    if request_id in scored_requests:
                        continue
                    # add to processing list
                    requests_to_score.append(request_id)
                
                # process requests
                if len(requests_to_score) > 0:
                    for request_id in requests_to_score:
                        output = generator_outputs[request_id]
                        score = reward_func(output)
                        scored_requests.append(request_id)
                        
        
        @rl.update_task(as_executable=False)
        async def train_model(model_config=model_config, workflow_ddict=workflow_ddict):
            return await trainer.train(model_config, workflow_ddict=workflow_ddict)

        @rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=128, operator=GREATER_THAN_THRESHOLD, as_executable=False)
        async def test_model():
            return await evaluate_func(model_config)

        # start generators
        for i in range(self.flow_config.num_generators):
            self._generator_tasks.append(generation_task(
                model_config=self.model_config,
                batch_size=self.flow_config.batch_size,
                _terminate_event=terminate_event,
                _workflow_ddict=workflow_ddict,
                _input_key=f"generator_{i}_input",
                _output_key=f"generator_{i}_output",
            ))
        #start scorers
        reward_funcs = self.trainer.reward_funcs
        reward_task_funcs = [reward_func for reward_func in reward_funcs if hasattr(reward_func, "_is_reward_task")]

        for i in range(self.flow_config.num_scorers):
            for reward_func in reward_task_funcs:
                self._scorer_tasks.append(scorer_task(
                    reward_func=reward_func,
                    _workflow_ddict=workflow_ddict,
                    _terminate_event=terminate_event,
                    _input_key=f"reward_{reward_func.__name__}_{i}_input",
                    _output_key=f"reward_{reward_func.__name__}_{i}_output",
                ))

        # start generator scheduler and gatherer 
        schedule_gather_fut = asyncio.gather(
            self._generation_schedule(workflow_ddict, terminate_event),
            self._generation_gather(workflow_ddict, terminate_event),
            self._scorer_schedule(workflow_ddict, terminate_event),
            self._scorer_gather(workflow_ddict, terminate_event),
        )

        async for state in rl.start():
            print(f"Iteration {state.iteration}: metric={state.metric_value}")

        return

 
