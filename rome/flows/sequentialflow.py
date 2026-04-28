from rome.workflow import Workflow
from rome.model import Model
from rome.trainer import Trainer

class SequentialFlow(Workflow):
    def __init__(
        self,
        flow: WorkflowEngine,
        trainer: Trainer,
        model: Model,
    ):
        super().__init__(flow=flow, trainer=trainer)
        self.model = model
        self.num_generators = 4
    
    async def launch(self):
        # first start generator tasks
        from dragon.data.ddict import DDict

        _generation_prompt_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
        _completion_ddict        = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
        _gentask_assignments_ddicts = []
        for i in range(self.num_generators):
            _gentask_assignments_ddicts.append(DDict(managers_per_node=1, n_nodes=1, total_mem=1024**3))

        @self.flow.function_task
        async def generator_task(_gentask_assignment_ddict, _completion_ddict, _stop_event, seed=42):
            import torch
            import numpy as np
            import random

            def set_seed(seed=seed) -> None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                np.random.seed(seed)
                random.seed(seed)

            _model, tokenizer = self.model._load_model_and_tokenizer()

            # build prompt set
            messages = [val for val in prompts for _ in range(num_samples_per_prompt)]
            
            # generate completions
            set_seed(seed)
            results = []
            processed_requests = []
            with torch.no_grad():
                ddict_keys = _gentask_assignment_dddict.get_keys()
                while not _stop_event.is_set():
                    for key in ddict_keys:
                        if key not in processed_requests:
                            request = _gentask_assignment_dddict[key]
                            prompt = request["prompt"]
                            num_samples_per_prompt = request["batch_size"]
                            batch = prompt * num_samples_per_prompt
                            inputs = tokenizer.apply_chat_template(
                                batch,
                                add_generation_prompt=True,
                                tokenize=True,
                                padding=True,
                                return_tensors="pt",
                            ).to(_model.device)
                            outputs = _model.generate(
                                inputs,
                                max_new_tokens=1024,
                                output_scores=True,
                                return_dict_in_generate=True,
                                do_sample=True,
                                top_p=0.95,
                                temperature=0.8,
                                pad_token_id = tokenizer.eos_token_id
                            )
                            transition_scores = _model.compute_transition_scores(
                                outputs.sequences,
                                outputs.scores,
                                normalize_logits=True  # applies log_softmax internally
                            )
                            processed_requests.append(key)
                            _completion_ddict[key] = {
                                "outputs": outputs.sequences.tolist(),
                                "transition_scores": transition_scores.tolist(),
                            }
                    await asyncio.sleep(1)  # yield control to event loop to allow other tasks to run
            return results
        
        @self.flow.function_task
        async def trainer_task():
            self.trainer.run_training(*args, **kwargs)
            return

