from typing import Callable
from enum import Enum

class TrainerStatus(Enum):
    """Enum class for managing the status of training tasks in the ROME framework."""
    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4
    NOT_ENOUGH_DATA = 5
    TRAINING_COMPLETE = 6

class Trainer:
    """Training manager in the ROME framework."""
    def __init__(self, manager: 'Manager'):
        self.manager = manager
        self.status = TrainerStatus.NOT_STARTED
        self.trainer_ddict = DDict()
        self.stop_event = Event()
        self.listener_fut = None

    def start(self, ):
        """Start the ROME training manager."""
        self.status = STARTING
        
        async def trainer_listener():
            #set status
            while not self.stop_event.is_set():
                self.training_func()
                
        
        listener_fut = trainer_listener()
        
        
        

    def stop(self):
        """Stop the ROME training manager."""
        self.stop_event.set()

    def status(self):
        return self.status