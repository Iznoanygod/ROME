from dataclasses import dataclass, field

@dataclass
class LoRAConfig:
    """LoRA / PEFT adapter knobs applied during model load.
    """

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

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
    lora_config: Optional[LoRAConfig] = None
    generation_config: Optional[GenerationConfig] = None
    dtype: Optional[str] = "auto"
    required_gpus: int = 1
    device_map: Any = "auto"
    max_seq_length: int = 2048
