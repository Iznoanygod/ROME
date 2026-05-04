from dataclasses import dataclass, field

@dataclass
class LoRAConfig:
    """LoRA / PEFT adapter knobs applied during model load.

    When ``enabled=False`` the model is loaded for full fine-tuning and
    no PEFT wrapping is performed.
    """

    enabled: bool = True
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
    """Base model selection and load-time settings.

    ``name`` is a HuggingFace hub ID or local path. The ``Trainer`` resolves
    the actual path-to-load each iteration via ``resolve_model_path`` so
    iteration N loads from iteration N-1's checkpoint when one exists.
    """
    base_model_name: Optional[str] = None
    model_name: Optional[str] = None
    lora_name: Optional[str] = None
    lora_config: LoRAConfig = None
    generation_config: Optional[GenerationConfig] = None
    dtype: Optional[str] = "auto"
    required_gpus: int = 1
    device_map: Any = "auto"
    max_seq_length: int = 2048
