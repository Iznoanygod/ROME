from rome.flows.sequentialflow import SequentialFlowConfig


def test_defaults():
    cfg = SequentialFlowConfig()
    assert cfg.iterations == 10
    assert cfg.reward_threshold is None
    
    assert cfg.operator is not None
    assert cfg.num_generators == 2
    assert cfg.num_scorers == 2
    assert cfg.batch_size == 4


def test_overrides():
    cfg = SequentialFlowConfig(
        iterations=100,
        reward_threshold=0.9,
        operator="custom",
        num_generators=8,
        num_scorers=4,
        batch_size=16,
    )
    assert cfg.iterations == 100
    assert cfg.reward_threshold == 0.9
    assert cfg.operator == "custom"
    assert cfg.num_generators == 8
    assert cfg.num_scorers == 4
    assert cfg.batch_size == 16