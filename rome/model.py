from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Model:
    base_model_name: str
    model_name: Optional[str] = None
    lora_name: Optional[str] = None
    dtype: Optional[str] = "auto"
    required_gpus: int = 1
    device_map: Any = "auto"

    def _load_model_and_tokenizer(self, inference_mode=False,dtype="auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from peft import PeftModel, get_peft_model, LoraConfig
        except Exception:
            PeftModel = None
            get_peft_model = None
            LoraConfig = None

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, padding_side="left", use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        if self.model_name:
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map, dtype=dtype)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, device_map=self.device_map, dtype=dtype)
        
        # If a LoRA checkpoint directory exists, attach it to the base model.
        if self.lora_id and os.path.isdir(self.lora_id) and any(Path(self.lora_id).iterdir()):
            model = PeftModel.from_pretrained(base_model, self.lora_id, is_trainable=True)
        else:
            if get_peft_model and LoraConfig:
                lora_conf = LoraConfig(
                    r=128, lora_alpha=256, lora_dropout=0.05, inference_mode=inference_mode, bias="none", task_type="CAUSAL_LM"
                )
                model = get_peft_model(base_model, lora_conf)
            else:
                model = base_model

        return model, tokenizer