import os

from rome.config import ModelConfig

WEIGHT_VERSION_KEY = "model_weight_version"

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

def save_model(model, model_config: ModelConfig) -> str:
    """Persist the model's weights for cross-task reload.

    Branches on :class:`ModelConfig`:

    * If ``lora_name`` is set, only the PEFT adapter is saved (the
      lightweight path — small files, fast to reload).
    * Otherwise the full model is saved to ``model_name``.

    Returns the directory the weights were written to. Raises
    ``ValueError`` if neither path is configured.
    """
    if model_config.lora_name is not None:
        model.save_pretrained(model_config.lora_name)
        return model_config.lora_name

    if model_config.model_name is None:
        raise ValueError(
            "save_model requires either lora_name (adapter-only save) "
            "or model_name (full-model save) on ModelConfig."
        )
    model.save_pretrained(model_config.model_name)
    return model_config.model_name

def reload_lora(model, model_config: ModelConfig) -> None:
    """Reload PEFT adapter weights into ``model`` in place.

    The existing wrapped model is kept and only the adapter state dict is
    swapped — the GPU tensors backing the base model are not reallocated.
    Raises ``ValueError`` if ``lora_name`` isn't set and ``FileNotFoundError``
    if the adapter directory doesn't exist yet.
    """
    if model_config.lora_name is None:
        raise ValueError("reload_lora requires lora_name on ModelConfig")
    if not os.path.exists(model_config.lora_name):
        raise FileNotFoundError(
            f"adapter not found at {model_config.lora_name}"
        )
    # Public PEFT helpers; importing lazily so the module stays importable in
    # environments where peft isn't installed (test conftest stubs it).
    from peft import set_peft_model_state_dict
    try:
        from peft import load_peft_weights
    except ImportError:
        from peft.utils import load_peft_weights

    state_dict = load_peft_weights(model_config.lora_name)
    set_peft_model_state_dict(model, state_dict)


def reload_model(model, model_config: ModelConfig) -> None:
    """In-place weight reload.

    Dispatches to :func:`reload_lora` when ``lora_name`` is set; otherwise
    loads the full model state dict from ``model_name`` into the existing
    model object (so callers keep their reference and any compiled graphs).
    """
    if model_config.lora_name is not None:
        reload_lora(model, model_config)
        return

    if model_config.model_name is None:
        raise ValueError(
            "reload_model requires either lora_name or model_name on ModelConfig"
        )
    if not os.path.exists(model_config.model_name):
        raise FileNotFoundError(
            f"model weights not found at {model_config.model_name}"
        )
    from transformers import AutoModelForCausalLM

    src = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    # Copy state into the existing model so we don't reallocate GPU tensors
    # or invalidate the caller's reference.
    model.load_state_dict(src.state_dict())
    del src


def bump_weight_version(workflow_ddict) -> int:
    """Increment the weight-version counter generators poll for reload signaling.

    Returns the new version. Single-writer (the trainer) and many-reader (the
    generator tasks); no compare-and-swap is needed because only the trainer
    bumps and DDict writes are atomic per-key.
    """
    new = read_weight_version(workflow_ddict) + 1
    workflow_ddict[WEIGHT_VERSION_KEY] = new
    return new

def read_weight_version(workflow_ddict, default: int = 0) -> int:
    """Read the current weight version, returning ``default`` when missing."""
    try:
        value = workflow_ddict[WEIGHT_VERSION_KEY]
    except (KeyError, TypeError):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default