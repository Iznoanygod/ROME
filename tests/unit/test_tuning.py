"""Unit tests for rome.tuning."""
from __future__ import annotations

import pytest

from rome.tuning import (
    BatchSizeController,
    BatchTuningConfig,
    ThroughputMonitor,
    ThroughputSample,
)


# ---------- ThroughputSample ----------

def test_sample_throughput_excludes_wait():
    s = ThroughputSample(batch_size=8, items=16, duration_s=2.0, wait_s=4.0)
    assert s.throughput == pytest.approx(8.0)


def test_sample_effective_throughput_includes_wait():
    s = ThroughputSample(batch_size=8, items=16, duration_s=2.0, wait_s=2.0)
    assert s.effective_throughput == pytest.approx(4.0)


def test_sample_throughput_zero_duration_is_zero():
    s = ThroughputSample(batch_size=8, items=16, duration_s=0.0, wait_s=1.0)
    assert s.throughput == 0.0


def test_sample_effective_throughput_zero_total_is_zero():
    s = ThroughputSample(batch_size=8, items=16, duration_s=0.0, wait_s=0.0)
    assert s.effective_throughput == 0.0


# ---------- ThroughputMonitor ----------

def test_monitor_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        ThroughputMonitor(window_size=0)


def test_monitor_record_and_read_all():
    m = ThroughputMonitor(window_size=8)
    m.record(ThroughputSample(batch_size=4, items=4, duration_s=1.0, task="scorer"))
    m.record(ThroughputSample(batch_size=4, items=8, duration_s=1.0, task="generator"))
    assert len(m.samples()) == 2
    assert len(m.samples("scorer")) == 1
    assert len(m.samples("generator")) == 1
    assert m.samples("missing") == []


def test_monitor_respects_window_per_task():
    m = ThroughputMonitor(window_size=3)
    for i in range(5):
        m.record(
            ThroughputSample(batch_size=4, items=i, duration_s=1.0, task="scorer")
        )
    samples = m.samples("scorer")
    # only the last 3 survive
    assert [s.items for s in samples] == [2, 3, 4]


def test_monitor_samples_for_batch_filters_per_arm():
    m = ThroughputMonitor(window_size=16)
    m.record(ThroughputSample(batch_size=4, items=4, duration_s=1.0, task="scorer"))
    m.record(ThroughputSample(batch_size=8, items=8, duration_s=1.0, task="scorer"))
    m.record(ThroughputSample(batch_size=4, items=4, duration_s=1.0, task="scorer"))

    arm4 = m.samples_for_batch(4, task="scorer")
    arm8 = m.samples_for_batch(8, task="scorer")
    assert len(arm4) == 2
    assert len(arm8) == 1


def test_monitor_means_handle_empty():
    m = ThroughputMonitor()
    assert m.mean_throughput() == 0.0
    assert m.mean_effective_throughput() == 0.0
    assert m.mean_wait() == 0.0


def test_monitor_means_compute_per_task():
    m = ThroughputMonitor()
    # scorer: 2 samples, throughput 4 and 8 -> mean 6
    m.record(ThroughputSample(batch_size=4, items=4, duration_s=1.0, wait_s=1.0, task="scorer"))
    m.record(ThroughputSample(batch_size=8, items=8, duration_s=1.0, wait_s=3.0, task="scorer"))
    # generator sample shouldn't pollute scorer mean
    m.record(ThroughputSample(batch_size=1, items=1, duration_s=1.0, wait_s=0.0, task="generator"))

    assert m.mean_throughput("scorer") == pytest.approx(6.0)
    assert m.mean_wait("scorer") == pytest.approx(2.0)
    # effective: 4/(1+1)=2 and 8/(1+3)=2 -> 2.0
    assert m.mean_effective_throughput("scorer") == pytest.approx(2.0)


def test_monitor_reset_clears_everything():
    m = ThroughputMonitor()
    m.record(ThroughputSample(batch_size=4, items=4, duration_s=1.0, task="scorer"))
    m.reset()
    assert m.samples() == []


# ---------- BatchTuningConfig ----------

def test_tuning_config_defaults():
    c = BatchTuningConfig()
    assert c.enabled is False
    assert c.objective == "effective"
    assert c.min_batch_size == 1
    assert c.max_batch_size == 64
    assert c.initial_batch_size == 4
    assert c.window_size == 64


def test_tuning_config_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        BatchTuningConfig(min_batch_size=0)
    with pytest.raises(ValueError):
        BatchTuningConfig(min_batch_size=8, max_batch_size=4)


def test_tuning_config_rejects_initial_out_of_bounds():
    with pytest.raises(ValueError):
        BatchTuningConfig(min_batch_size=4, max_batch_size=16, initial_batch_size=2)
    with pytest.raises(ValueError):
        BatchTuningConfig(min_batch_size=4, max_batch_size=16, initial_batch_size=32)


def test_tuning_config_rejects_unknown_objective():
    with pytest.raises(ValueError):
        BatchTuningConfig(objective="something_else")


def test_tuning_config_auto_candidates_geometric():
    c = BatchTuningConfig(min_batch_size=1, max_batch_size=16, initial_batch_size=4)
    assert c.candidates() == [1, 2, 4, 8, 16]


def test_tuning_config_auto_candidates_includes_max_even_when_not_power_of_two():
    c = BatchTuningConfig(min_batch_size=1, max_batch_size=10, initial_batch_size=4)
    cands = c.candidates()
    assert cands[0] == 1
    assert cands[-1] == 10
    # strictly increasing, deduplicated
    assert cands == sorted(set(cands))


def test_tuning_config_auto_candidates_min_equals_max():
    c = BatchTuningConfig(min_batch_size=4, max_batch_size=4, initial_batch_size=4)
    assert c.candidates() == [4]


def test_tuning_config_explicit_candidates_passthrough():
    c = BatchTuningConfig(
        min_batch_size=1,
        max_batch_size=64,
        initial_batch_size=4,
        candidate_batch_sizes=[2, 4, 8, 16],
    )
    assert c.candidates() == [2, 4, 8, 16]


# ---------- BatchSizeController interface ----------

def test_controller_propose_is_abstract():
    ctrl = BatchSizeController(BatchTuningConfig(), ThroughputMonitor())
    with pytest.raises(NotImplementedError):
        ctrl.propose(task="scorer", current=4)
