from rome.flows.streamflow import StreamFlowConfig
from rome.tuning import BatchTuningConfig


def test_defaults():
    cfg = StreamFlowConfig()
    assert cfg.iterations == 10
    assert cfg.reward_threshold is None
    assert cfg.operator is not None
    assert cfg.num_generators == 2
    assert cfg.num_scorers == 2
    assert cfg.scorer_batch_size == 4
    # tuning defaults to a disabled tuner, not None — flows can always read it
    assert isinstance(cfg.tuning, BatchTuningConfig)
    assert cfg.tuning.enabled is False


def test_overrides():
    tuning = BatchTuningConfig(
        enabled=True,
        min_batch_size=2,
        max_batch_size=32,
        initial_batch_size=8,
        window_size=16,
    )
    cfg = StreamFlowConfig(
        iterations=100,
        reward_threshold=0.9,
        operator="custom",
        num_generators=8,
        num_scorers=4,
        scorer_batch_size=16,
        tuning=tuning,
    )
    assert cfg.iterations == 100
    assert cfg.reward_threshold == 0.9
    assert cfg.operator == "custom"
    assert cfg.num_generators == 8
    assert cfg.num_scorers == 4
    assert cfg.scorer_batch_size == 16
    assert cfg.tuning is tuning
