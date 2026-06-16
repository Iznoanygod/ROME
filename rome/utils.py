import os
from typing import Optional

from rome.config import ModelConfig

WEIGHT_VERSION_KEY = "model_weight_version"
WEIGHT_PATH_KEY = "model_checkpoint_path"


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


def save_model(model, model_config: ModelConfig, path: Optional[str] = None) -> str:
    """Persist the model's weights for cross-task reload.

    When ``path`` is given, the weights are written there verbatim — used by
    the streaming sync callback to publish per-step versioned subdirs. When
    ``path`` is ``None``, falls back to in-place writes at
    ``model_config.lora_name`` (LoRA adapter only) or ``model_config.model_name``
    (full model). Returns the directory the weights were written to.
    """
    if path is not None:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        return path

    if model_config.lora_name is not None:
        model.save_pretrained(model_config.lora_name)
        return model_config.lora_name

    if model_config.model_name is None:
        raise ValueError(
            "save_model requires either a path, lora_name (adapter-only save), "
            "or model_name (full-model save) on ModelConfig."
        )
    model.save_pretrained(model_config.model_name)
    return model_config.model_name


def reload_lora(model, model_config: ModelConfig, path: Optional[str] = None) -> None:
    """Reload PEFT adapter weights into ``model`` in place.

    Reads from ``path`` when provided (the unified DDict-published checkpoint
    path), otherwise from ``model_config.lora_name``. The existing wrapped
    model is kept and only the adapter state dict is swapped — the GPU
    tensors backing the base model are not reallocated.
    """
    src = path if path is not None else model_config.lora_name
    if src is None:
        raise ValueError("reload_lora requires path or lora_name on ModelConfig")
    if not os.path.exists(src):
        raise FileNotFoundError(f"adapter not found at {src}")

    from peft import set_peft_model_state_dict
    try:
        from peft import load_peft_weights
    except ImportError:
        from peft.utils import load_peft_weights

    state_dict = load_peft_weights(src)
    set_peft_model_state_dict(model, state_dict)


def reload_model(model, model_config: ModelConfig, path: Optional[str] = None) -> None:
    """In-place weight reload.

    Dispatches to :func:`reload_lora` when ``lora_name`` is set; otherwise
    loads the full model state dict from ``path`` (or ``model_name``) into
    the existing model object so callers keep their reference and any
    compiled graphs.
    """
    if model_config.lora_name is not None:
        reload_lora(model, model_config, path)
        return

    src = path if path is not None else model_config.model_name
    if src is None:
        raise ValueError(
            "reload_model requires path, lora_name, or model_name on ModelConfig"
        )
    if not os.path.exists(src):
        raise FileNotFoundError(f"model weights not found at {src}")
    from transformers import AutoModelForCausalLM

    new = AutoModelForCausalLM.from_pretrained(src)
    model.load_state_dict(new.state_dict())
    del new


def bump_weight_version(workflow_ddict, path: Optional[str] = None) -> int:
    """Increment the weight-version counter generators poll for reload signaling.

    When ``path`` is given, also publishes it under :data:`WEIGHT_PATH_KEY` so
    generators can find the new checkpoint without convention. Returns the
    new version. Single-writer (the trainer) and many-reader (the generator
    tasks); no compare-and-swap is needed because only the trainer bumps and
    DDict writes are atomic per-key.
    """
    new = read_weight_version(workflow_ddict) + 1
    workflow_ddict[WEIGHT_VERSION_KEY] = new
    if path is not None:
        workflow_ddict[WEIGHT_PATH_KEY] = path
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


def read_weight_path(workflow_ddict) -> Optional[str]:
    """Read the latest published checkpoint path, or ``None`` if unset."""
    try:
        value = workflow_ddict[WEIGHT_PATH_KEY]
    except (KeyError, TypeError):
        return None
    return value if value else None


def maybe_reload_weights(model, model_config: ModelConfig, workflow_ddict, local_version: int):
    """Reload weights into ``model`` if the trainer has published a newer version.

    Shared by every flow's generator task. Returns ``(model, new_local_version)``.
    Only advances ``local_version`` when a reload was actually performed, so a
    bump with no path published doesn't silently swallow a future reload.
    """
    remote_version = read_weight_version(workflow_ddict)
    if remote_version <= local_version:
        return model, local_version
    path = read_weight_path(workflow_ddict)
    if path is None and model_config.lora_name is None and model_config.model_name is None:
        return model, local_version
    reload_model(model, model_config, path)
    return model, remote_version


def _trainer_callback_base():
    try:
        from transformers import TrainerCallback
        return TrainerCallback
    except ImportError:
        return object


class WeightSyncCallback(_trainer_callback_base()):
    """Persist weights and bump the version counter during training.

    Used by :class:`StreamFlow` to publish a fresh checkpoint every
    ``interval`` trainer steps. Generators poll
    :data:`WEIGHT_VERSION_KEY` between batches and reload from
    :data:`WEIGHT_PATH_KEY` when they see a newer version.

    Parameters
    ----------
    workflow_ddict
        The shared DDict the flow uses to coordinate tasks.
    model_config : ModelConfig
        Model config — forwarded to :func:`save_model` so the same
        save_pretrained branching is used everywhere.
    checkpoint_dir : str
        Directory to write versioned subdirs into
        (``{checkpoint_dir}/step_{version}``).
    interval : int, optional
        Minimum number of trainer steps between checkpoints. Default 1.
    """

    def __init__(self, workflow_ddict, model_config: ModelConfig, checkpoint_dir: str, interval: int = 1):
        self._workflow_ddict = workflow_ddict
        self._model_config = model_config
        self._checkpoint_dir = checkpoint_dir
        self._interval = max(1, interval)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self._interval != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        version = read_weight_version(self._workflow_ddict) + 1
        path = os.path.join(self._checkpoint_dir, f"step_{version}")
        save_model(model, self._model_config, path)
        bump_weight_version(self._workflow_ddict, path)
        return control
