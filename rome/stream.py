from typing import Callable
from enum import Enum
from asyncio import awaitable

from dragon.data.ddict import DDict
from dragon.native.event import Event


class StreamStatus(Enum):
    """Enum class for managing the status of inference stream tasks in the ROME framework."""
    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    RELOADING = 3
    STOPPING = 4
    STOPPED = 5

class StreamConfig:
    """Configuration class for managing inference stream settings in the ROME framework."""
    def __init__(self, model_path: str, num_streams: int = 1, num_gpus: int = 1, stream_func: Callable = None):
        self.model_path = model_path
        self.num_streams = num_streams
        self.num_gpus = num_gpus
        self.stream_func = stream_func

class StreamTask:
    """StreamTask class for managing individual inference stream tasks in the ROME framework."""
    def __init__(self, task_fut: awaitable, config: StreamConfig):
        self.task_fut = task_fut
        self.config = config
        self.task_ddict = DDict()
        task_ddict['status'] = StreamStatus.NOT_STARTED
        self.reload_event = Event()
        self.stop_event = Event()

class Stream:
    """Stream class for managing inference stream task in the ROME framework."""
    def __init__(self, manager: 'Manager'):
        self.manager = manager
        self.asyncflow = manager.asyncflow
        self.stream_tasks = []
        self.listener_fut = None

    def start(self, config: StreamConfig):
        """Start the inference stream tasks."""
        stream_func = config.stream_func
        for i in range(config.num_streams):
            task_fut = stream_func()
            stream_task = StreamTask(task_fut, config)
            stream_task.task_ddict['status'] = StreamStatus.STARTING
            self.stream_tasks.append(stream_task)

    def reload_model(self, model_path: str, wait_for_reload: bool = True):
        """Signal the manager to reload the model for inference stream tasks."""
        for task in self.stream_tasks:
            task.config.model_path = model_path
            task.reload_event.set()
        if wait_for_reload:
            for task in self.stream_tasks:
                while task.reload_event.is_set():
                    time.sleep(1)

    def get_status(self):
        """Get the current status of the inference stream tasks."""
        return [task.task_ddict['status'] for task in self.stream_tasks]

    def stop(self, wait_for_stop: bool = True):
        """Signal the manager to stop the inference stream tasks."""
        for task in self.stream_tasks:
            task.stop_event.set()
        if wait_for_stop:
            for task in self.stream_tasks:
                while task.task_ddict['status'] != StreamStatus.STOPPED:
                    time.sleep(1)

    