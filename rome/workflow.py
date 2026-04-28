import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple
from radical.asyncflow import WorkflowEngine 
from rome.trainer import Trainer
from rome.model import Model
import itertools

logger = logging.getLogger(__name__)

class Workflow:
    def __init__(
        self,
        flow: WorkflowEngine,
        trainer: Trainer,
        model: Model,
    ):
        self.flow = flow
        self.trainer = trainer
        self.model = model
    
    async def launch(self):
        raise NotImplementedError("Subclasses of Workflow must implement launch()")