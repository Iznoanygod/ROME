import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple

from rhapsody.backends import DragonExecutionBackendV3
from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node
from dragon.native.event import Event
from radical.asyncflow import WorkflowEngine
import itertools

logger = logging.getLogger(__name__)
class GPUAllocatorRoundRobin:
    """Small helper to round-robin individual GPUs across nodes."""

    def __init__(self, node_gpu_list: List[Tuple[str, List[int]]]):
        self._flat = []
        for node, gpus in node_gpu_list:
            for g in gpus:
                self._flat.append((node, [g]))
        self._it = itertools.cycle(self._flat) if self._flat else iter([])
    
    def next(self) -> Tuple[str, List[int]]:
        return next(self._it)

class Workflow:
    def __init__(
        self,
        flow: Optional[WorkflowEngine] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        prompt_gen_batch_size: int = 2,
    ):
        self.backend_config = backend_config or {}
        self.flow: Optional[WorkflowEngine] = flow
        self._backend = None
        self.prompt_gen_batch_size = prompt_gen_batch_size

        # placeholders for discovered resources
        self.nodes_with_gpus: List[Tuple[str, List[int]]] = []
        self.gpu_allocator: Optional[GPUAllocatorRoundRobin] = None

        # events used across tasks
        self._terminate = asyncio.Event()
        self._iteration_reset = Event()
    def find_gpus(self) -> List[Tuple[str, List[int]]]:
        """Discover nodes and GPUs via dragon System/Node API."""
        nodes = []
        sys = System()
        for huid in sys.nodes:
            node = Node(huid)
            nodes.append((node.hostname, list(node.gpus)))
        self.nodes_with_gpus = nodes
        return nodes

    def build_gpu_allocator(self):
        if not self.nodes_with_gpus:
            self.find_gpus()
        self.gpu_allocator = GPUAllocatorRoundRobin(self.nodes_with_gpus)

    async def generation_listener(self, poll_interval: float = 1.0):
        if not self.generate_task or not self.gpu_allocator:
            raise RuntimeError("generate_task or gpu_allocator not configured")
        running = []
        while not self._terminate.is_set():
            if self._iteration_reset.is_set():
                # reset logic for each training iteration
                running = []
                await asyncio.sleep(poll_interval)
                continue

            for fam, needed in list(self.superfamily_gen_ddict.items()):
                if fam in self.generated_families:
                    continue
                gpu = self.gpu_allocator.next()
                policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=gpu[0], gpu_affinity=gpu[1])
                # schedule the task; caller's generate_task must accept task_description param
                fut = self.generate_task(task_description={"process_template": {"policy": policy}})
                running.append(fut)
                self.generated_families[fam] = True
            await asyncio.sleep(poll_interval)
        if running:
            await asyncio.gather(*running)
    async def scoring_listener(self, poll_interval: float = 1.0):
        """Monitor folded_families and submit scoring tasks."""
        if not self.score_task or not self.gpu_allocator:
            raise RuntimeError("score_task or gpu_allocator not configured")
        running = []
        while not self._terminate.is_set():
            if self._iteration_reset.is_set():
                running = []
                await asyncio.sleep(poll_interval)
                continue
            for fam in list(self.folded_families.keys()):
                gpu = self.gpu_allocator.next()
                policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=gpu[0], gpu_affinity=gpu[1])
                fut = self.score_task(task_description={"process_template": {"policy": policy}})
                running.append(fut)
            await asyncio.sleep(poll_interval)
        if running:
            await asyncio.gather(*running)

    async def launch_trainer(self, host_name: str, policy: Optional[Policy] = None):
        """
        Launch the trainer task (flow.function_task). The trainer_task must be registered via register_task_providers.
        This method returns the task future. The trainer should observe self._iteration_reset Event or accept reset_event.
        """
        if not self.trainer_task:
            raise RuntimeError("trainer_task not registered")
        if policy is None:
            # default policy targeting host_name
            policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=host_name, gpu_affinity=[0])
        task_description = {"process_template": {"policy": policy}}
        fut = self.trainer_task(self._iteration_reset, task_description=task_description)
        return fut
    
    async def launch(self, host_for_trainer: Optional[str] = None):
        self.find_gpus()
        self.build_gpu_allocator()
        gen_task = asyncio.create_task(self.generation_listener())
        score_task = asyncio.create_task(self.scoring_listener())

        trainer_host = host_for_trainer or (self.nodes_with_gpus[0][0] if self.nodes_with_gpus else "localhost")
        trainer_fut = await self.launch_trainer(host_name=trainer_host)

        try:
            await trainer_fut
        finally:
            # trigger termination and wait for listeners
            self._terminate.set()
            await asyncio.gather(gen_task, score_task, return_exceptions=True)
            # flow shutdown handled externally by caller (or add here:
            if self.flow:
                await self.flow.shutdown()