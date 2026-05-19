from rome.config import LoRAConfig, ModelConfig


def test_loraconfig_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 16
    assert cfg.alpha == 16
    assert cfg.dropout == 0.0
    assert cfg.bias == "none"
    assert cfg.target_modules == [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_loraconfig_target_modules_not_shared():
    a = LoRAConfig()
    b = LoRAConfig()
    a.target_modules.append("extra")
    assert "extra" not in b.target_modules


def test_loraconfig_overrides():
    cfg = LoRAConfig(r=64, alpha=128, dropout=0.1, bias="all")
    assert (cfg.r, cfg.alpha, cfg.dropout, cfg.bias) == (64, 128, 0.1, "all")


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