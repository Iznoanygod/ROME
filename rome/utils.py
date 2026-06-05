import os

from rome.config import ModelConfig

def load_model(model_config: ModelConfig):
    """Load model and tokenizer according to the provided model configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load tokenizer from base model if that is set, if not load tokenizer from model_name\
    if model_config.base_model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name, padding_side="left")
    elif model_config.model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, padding_side="left")
    else:
        raise ValueError("Either base_model_name or model_name must be set in ModelConfig.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load model, base_model if no model_name, or load model_name is that is given
    if model_config.model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.base_model_name)

    # creating lora model 
    if model_config.lora_name is not None:
        from peft import get_peft_model, LoraConfig, PeftModel
        
        # if lora model exists, load that, otherwise create new lora
        if os.path.exists(model_config.lora_name):
            model = PeftModel.from_pretrained(model, model_config.lora_name, is_trainable=True)

        else:
            lora_config = model_config.lora_config
            if lora_config is None:
                lora_config = LoraConfig(
                    r=128,
                    lora_alpha=128,
                    lora_dropout=0.0,
                    inference_mode=False,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)
    return model, tokenizer

def reload_lora(model, model_config: ModelConfig, checkpoint_path: str):
    """Hot-reload a LoRA adapter checkpoint into an already-loaded PEFT model.

    Used by streaming workflows to swap in a freshly trained adapter
    between inference batches without tearing down the base model.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return model
    if hasattr(model, "load_adapter"):
        try:
            model.delete_adapter("default")
        except (KeyError, ValueError, AttributeError):
            pass
        model.load_adapter(checkpoint_path, adapter_name="default")
    return model


def save_model(model, model_config: ModelConfig):
    pass

