@dataclass
class ModelConfig:
    """Model configurations used for loading the model and tokenizer, as well as generation.

    Parameters
    ----------
    base_model_name: Optional[str]
        Foundation model name, used to load tokenizer and model weights(when model_name is not provided). 
        Should be a model name recognized by HuggingFace's AutoModelForCausalLM and AutoTokenizer classes.
    model_name: Optional[str]
        Full model name, used to load model weights. If no model_name is provided, base_model_name is used.
    lora_name: Optional[str]
        LoRA adapter name, used to load LoRA adapter weights. If no lora_name is provided, no 
    """
    base_model_name: Optional[str] = None
    model_name: Optional[str] = None
    lora_name: Optional[str] = None
    lora_config: Optional[LoraConfig] = None
    generation_config: Optional[GenerationConfig] = None
    dtype: Optional[str] = "auto"
    required_gpus: int = 1
    device_map: Any = "auto"
    max_seq_length: int = 2048

class GRPOConfig:
    """Configuration class for managing GRPO training settings in the ROME framework."""
    def __init__(self, model_path: str, learning_rate: float = 0.001, batch_size: int = 32, num_epochs: int = 10):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        

class GRPOTrainer:
    """GRPO trainer in the ROME framework."""
    def train(self, GRPOConfig, **kwargs):
