from rome.workflow import Workflow
from rose.metrics import GREATER_THAN_THRESHOLD

class StreamFlowConfig(Workflow):
    """Configuration for StreamFlow."""
    def __init__(
        self,
        iterations: Optional[int] = 10,
        reward_threshold: Optional[float] = None,
        operator: Optional[str] = GREATER_THAN_THRESHOLD,
        num_generators: int = 2,
        num_scorers: int = 2,
    ):
        self.reward_threshold = reward_threshold
        self.operator = operator