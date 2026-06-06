"""Unit tests for the StreamFlow scaffold.

Covers the contract the future UCB controller and streaming loop will rely
on: dynamic batch-size publish/read through the shared ddict and throughput
sampling through the monitor.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from rome.config import ModelConfig
from rome.flows.streamflow import StreamFlow, StreamFlowConfig
from rome.trainer import Trainer
from rome.tuning import BatchTuningConfig


def _make_flow(*, tuning: BatchTuningConfig | None = None) -> StreamFlow:
    trainer = Trainer(gpus=1, reward_funcs=[])
    return StreamFlow(
        model_config=ModelConfig(),
        trainer=trainer,
        evaluate_func=lambda mc: 0.0,
        asyncflow=object(),
        flow_config=StreamFlowConfig(tuning=tuning),
    )


def test_constructor_creates_monitor_with_config_window():
    flow = _make_flow(tuning=BatchTuningConfig(window_size=7))
    assert flow.monitor.window_size == 7
    assert flow._scorer_tasks == []
    assert flow._generator_tasks == []


def test_initial_scorer_batch_size_uses_static_when_tuning_disabled():
    flow = _make_flow()
    flow.flow_config.scorer_batch_size = 5
    assert flow.initial_scorer_batch_size() == 5


def test_initial_scorer_batch_size_uses_tuning_when_enabled():
    tuning = BatchTuningConfig(enabled=True, initial_batch_size=8)
    flow = _make_flow(tuning=tuning)
    flow.flow_config.scorer_batch_size = 99  # ignored when tuning enabled
    assert flow.initial_scorer_batch_size() == 8


def test_read_scorer_batch_size_returns_default_when_missing():
    flow = _make_flow()
    assert flow.read_scorer_batch_size({}, default=4) == 4


def test_read_scorer_batch_size_returns_published_value():
    flow = _make_flow()
    ddict = {}
    flow.publish_scorer_batch_size(ddict, 16)
    assert flow.read_scorer_batch_size(ddict, default=4) == 16


def test_read_scorer_batch_size_falls_back_on_unparseable():
    flow = _make_flow()
    ddict = {StreamFlow.BATCH_SIZE_KEY: "not-a-number"}
    assert flow.read_scorer_batch_size(ddict, default=4) == 4


def test_publish_scorer_batch_size_validates_bounds_when_tuning_enabled():
    tuning = BatchTuningConfig(
        enabled=True, min_batch_size=2, max_batch_size=8, initial_batch_size=4
    )
    flow = _make_flow(tuning=tuning)
    ddict = {}
    flow.publish_scorer_batch_size(ddict, 8)  # at upper bound — ok
    assert ddict[StreamFlow.BATCH_SIZE_KEY] == 8
    with pytest.raises(ValueError):
        flow.publish_scorer_batch_size(ddict, 16)
    with pytest.raises(ValueError):
        flow.publish_scorer_batch_size(ddict, 1)


def test_publish_scorer_batch_size_skips_validation_when_tuning_disabled():
    flow = _make_flow()  # tuning disabled by default
    ddict = {}
    # outside default bounds, but tuning is off so anything goes
    flow.publish_scorer_batch_size(ddict, 256)
    assert ddict[StreamFlow.BATCH_SIZE_KEY] == 256


def test_sample_scorer_records_sample_with_measured_duration():
    flow = _make_flow()
    with flow.sample_scorer(batch_size=4, items=4, wait_s=0.5):
        time.sleep(0.01)

    samples = flow.monitor.samples("scorer")
    assert len(samples) == 1
    s = samples[0]
    assert s.batch_size == 4
    assert s.items == 4
    assert s.wait_s == pytest.approx(0.5)
    assert s.duration_s > 0


def test_sample_scorer_records_even_on_exception():
    flow = _make_flow()
    with pytest.raises(RuntimeError):
        with flow.sample_scorer(batch_size=8, items=8):
            raise RuntimeError("boom")
    # sample still recorded so a flaky batch doesn't blind the controller
    assert len(flow.monitor.samples("scorer")) == 1


def test_launch_is_not_implemented_yet():
    flow = _make_flow()
    with pytest.raises(NotImplementedError):
        asyncio.run(flow.launch())
