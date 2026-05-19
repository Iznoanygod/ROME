from rome.config import ModelConfig


def test_modelconfig_defaults():
    cfg = ModelConfig()
    assert cfg.base_model_name is None
    assert cfg.model_name is None
    assert cfg.lora_name is None
    assert cfg.lora_config is None
    assert cfg.dtype == "auto"
    assert cfg.required_gpus == 1
    assert cfg.device_map == "auto"
    assert cfg.max_seq_length == 2048


def test_modelconfig_set_values():
    cfg = ModelConfig(
        base_model_name="meta-llama/Llama-3.2-1B",
        lora_name="my-adapter",
        required_gpus=4,
    )
    assert cfg.base_model_name == "meta-llama/Llama-3.2-1B"
    assert cfg.lora_name == "my-adapter"
    assert cfg.required_gpus == 4