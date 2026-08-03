from radical.asyncflow import WorkflowEngine
from dragon.data.ddict import DDict
from dragon.native.event import Event

class Manager:
    """Manager class for handling tasks and data in ROME framework."""
    
    def __init__(self, asyncflow: WorkflowEngine):
        self.asyncflow = asyncflow
        self.trainer = Trainer(self)
        self.manager_ddict = DDict()
        self.stream = Stream(self)
        self.stop_event = Event()

    def start(self):
        """Start the ROME manager."""

        """start ROME Stream"""
        self.stream.start()

        self.trainer.start()
        
        

    def stop(self):

    def get_training_status(self):
        """Get the status of the training tasks."""
        return self.trainer.status

    def add_training_data(self, prompt, completion, score):
        """Add training data to the manager."""