from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Model:
    model_name: str
    lora_name: Optional[str] = None
    dtype: Optional[str] = None
    required_gpus: int = 1
    local_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def _load_model_and_tokenizer(self, dtype="auto"):
        # lazy imports
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from peft import PeftModel, get_peft_model, LoraConfig
        except Exception:
            PeftModel = None
            get_peft_model = None
            LoraConfig = None

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        # If a LoRA checkpoint directory exists, attach it to the base model.
        if self.lora_id and os.path.isdir(self.lora_id) and any(Path(self.lora_id).iterdir()):
            base_model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device_map, dtype=dtype)
            model = PeftModel.from_pretrained(base_model, self.lora_id, is_trainable=True)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device_map, dtype=dtype)
            if get_peft_model and LoraConfig:
                lora_conf = LoraConfig(
                    r=128, lora_alpha=256, lora_dropout=0.05, inference_mode=False, bias="none", task_type="CAUSAL_LM"
                )
                model = get_peft_model(base_model, lora_conf)
            else:
                model = base_model

        return model, tokenizer