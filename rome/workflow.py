import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple
from radical.asyncflow import WorkflowEngine 
from rome.trainer import Trainer
import itertools

logger = logging.getLogger(__name__)

class Workflow:
    def __init__(
        self,
        flow: WorkflowEngine,
        trainer: Trainer,
    ):
        self.flow = flow
        self.trainer = trainer
    
    async def launch(self):
        raise NotImplementedError("Subclasses of Workflow must implement launch()")