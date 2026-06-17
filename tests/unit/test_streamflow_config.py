from rome.flows.streamflow import StreamFlowConfig


def test_defaults():
    cfg = StreamFlowConfig()
    assert cfg.iterations == 10
    assert cfg.reward_threshold is None
    assert cfg.operator is not None
    assert cfg.num_generators == 2
    assert cfg.num_scorers == 2
    assert cfg.batch_size == 4
    assert cfg.num_generations_per_prompt == 4
    assert cfg.prompts == []
    assert cfg.max_buffer_per_prompt == 32
    assert cfg.checkpoint_dir is None
    assert cfg.checkpoint_interval == 1


def test_overrides():
    cfg = StreamFlowConfig(
        iterations=100,
        reward_threshold=0.9,
        operator="custom",
        num_generators=8,
        num_scorers=4,
        batch_size=16,
        num_generations_per_prompt=8,
        prompts=["p1", "p2"],
        max_buffer_per_prompt=64,
        checkpoint_dir="/tmp/ckpts",
        checkpoint_interval=5,
    )
    assert cfg.iterations == 100
    assert cfg.reward_threshold == 0.9
    assert cfg.operator == "custom"
    assert cfg.num_generators == 8
    assert cfg.num_scorers == 4
    assert cfg.batch_size == 16
    assert cfg.num_generations_per_prompt == 8
    assert cfg.prompts == ["p1", "p2"]
    assert cfg.max_buffer_per_prompt == 64
    assert cfg.checkpoint_dir == "/tmp/ckpts"
    assert cfg.checkpoint_interval == 5


def test_prompts_copied_not_aliased():
    src = ["a", "b"]
    cfg = StreamFlowConfig(prompts=src)
    src.append("c")
    assert cfg.prompts == ["a", "b"]


def test_checkpoint_interval_clamped_at_construction_is_runtime_concern():
    # The clamp lives inside WeightSyncCallback; the config keeps the raw value.
    cfg = StreamFlowConfig(checkpoint_interval=0)
    assert cfg.checkpoint_interval == 0
