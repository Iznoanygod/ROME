"""Full mock workflow test for rome.flows.sequentialflow.SequentialFlow.

Exercises ``SequentialFlow.launch()`` end-to-end with a Dragon-style runtime
(DDict + Event), a faked ``radical.asyncflow`` engine, and a faked
``rose`` reinforcement learner. Every workflow task is replaced with a
placeholder that returns a preset value so the test can assert on the
orchestration without needing real models, real Dragon, or real GPUs.

This file runs under plain ``pytest`` because Dragon is *mocked*. A
real-Dragon variant cannot be launched the same way -- Dragon owns process
startup and must be invoked with its own launcher, e.g.::

    dragon -m pytest tests/integration/test_full_workflow_dragon.py

A real-runtime variant would also have to drop the FakeDDict / OneShotEvent
fixtures and import ``DDict`` / ``Event`` directly from ``dragon``.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from rome.config import ModelConfig
from rome.workflow import Workflow


# ---------------------------------------------------------------------------
# Dragon-style fakes
# ---------------------------------------------------------------------------

class FakeDDict(dict):
    """Dragon DDict stand-in. The real DDict behaves as a shared dict."""

    def __init__(self):
        super().__init__()
        # Pre-seed every queue/output bucket SequentialFlow.launch touches.
        # Real Dragon code does the same lazily; tests fill them up front.
        for k in (
            "generation_requests",
            "generator_outputs",
        ):
            self[k] = {}


class OneShotEvent:
    """Dragon Event stand-in: trips to "set" after `limit` polls."""

    def __init__(self, limit: int = 1):
        self._calls = 0
        self._limit = limit

    def is_set(self) -> bool:
        result = self._calls >= self._limit
        self._calls += 1
        return result

    def set(self) -> None:
        self._calls = self._limit


# ---------------------------------------------------------------------------
# asyncflow + rose fakes
# ---------------------------------------------------------------------------

class FakeAsyncflow:
    """Stands in for ``radical.asyncflow.WorkflowEngine``.

    ``function_task`` returns a wrapper that records the call and immediately
    yields control instead of running an infinite polling loop.
    """

    def __init__(self):
        self.submitted = []

    def function_task(self, fn):
        submitted = self.submitted

        def _spawn(*args, **kwargs):
            submitted.append({"fn": fn.__name__, "args": args, "kwargs": kwargs})
            # Schedule the coroutine but don't await it forever. With a
            # OneShotEvent the underlying loops exit after one tick.
            return asyncio.ensure_future(fn(*args, **kwargs))

        return _spawn


class FakeLearner:
    """Stands in for ``rose.learner.SequentialReinforcementLearner``.

    Captures decorated ``update_task`` and ``as_stop_criterion`` callbacks and
    drives them through one preset iteration via ``start()``.
    """

    def __init__(self, asyncflow=None):
        self._update_fn = None
        self._stop_fn = None
        self._stop_kwargs = {}
        self.iterations_yielded = []

    def update_task(self, **decorator_kwargs):
        def wrap(fn):
            self._update_fn = fn
            return fn
        return wrap

    def as_stop_criterion(self, **decorator_kwargs):
        self._stop_kwargs = decorator_kwargs

        def wrap(fn):
            self._stop_fn = fn
            return fn
        return wrap

    async def start(self):
        # Drive exactly one iteration. The launch() body decorates train_model
        # and test_model, so both must be invocable here.
        await self._update_fn()
        metric_value = await self._stop_fn()
        state = types.SimpleNamespace(iteration=0, metric_value=metric_value)
        self.iterations_yielded.append(state)
        yield state


# ---------------------------------------------------------------------------
# Fixture: patch dragon + rose to point at the fakes for this test only
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_runtime(monkeypatch):
    """Replace the dragon/rose imports in rome.flows.sequentialflow."""
    import rome.flows.sequentialflow as sf

    monkeypatch.setattr(sf, "DDict", FakeDDict)
    monkeypatch.setattr(sf, "Event", OneShotEvent)
    monkeypatch.setattr(sf, "SequentialReinforcementLearner",
                        lambda asyncflow=None: FakeLearner(asyncflow))
    return sf


# ---------------------------------------------------------------------------
# Preset-value placeholder trainer
# ---------------------------------------------------------------------------

class PlaceholderTrainer:
    """Mock Trainer. ``train`` returns a preset value instead of using TRL."""

    PRESET_TRAIN_RESULT = {"loss": 0.123, "step": 1}

    def __init__(self, reward_funcs):
        self._reward_funcs = list(reward_funcs)
        self.train_calls = []

    @property
    def reward_funcs(self):
        return self._reward_funcs

    async def train(self, model_config, workflow_ddict, **kwargs):
        self.train_calls.append({
            "model_config": model_config,
            "workflow_ddict": workflow_ddict,
            "kwargs": kwargs,
        })
        return self.PRESET_TRAIN_RESULT


# ---------------------------------------------------------------------------
# The actual full mock workflow test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_mock_workflow_runs_one_iteration(patched_runtime):
    """SequentialFlow.launch executes one full iteration against placeholders.

    Every moving piece is mocked:
      * Dragon DDict/Event -> in-process fakes
      * radical.asyncflow engine -> FakeAsyncflow (function_task records calls)
      * rose learner -> FakeLearner (drives one iteration)
      * Trainer.train -> preset {"loss": 0.123, "step": 1}
      * evaluate_func -> preset 0.99
      * Reward functions -> preset score 1.0
    """
    from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig

    PRESET_REWARD = 1.0
    PRESET_EVAL = 0.99

    @Workflow.reward_task
    def fake_reward(output):
        return PRESET_REWARD

    @Workflow.evaluate_task
    async def fake_evaluate(model_config):
        return PRESET_EVAL

    trainer = PlaceholderTrainer(reward_funcs=[fake_reward])
    asyncflow = FakeAsyncflow()

    flow = SequentialFlow(
        model_config=ModelConfig(base_model_name="fake/model"),
        trainer=trainer,
        evaluate_func=fake_evaluate,
        asyncflow=asyncflow,
        flow_config=SequentialFlowConfig(
            iterations=1,
            num_generators=2,
            num_scorers=2,
            batch_size=4,
        ),
    )

    # Sanity: the patched runtime swapped the learner type
    assert isinstance(flow.rl, FakeLearner)

    await flow.launch()

    # --- Generator tasks were spawned, one per generator slot -------------
    gen_calls = [c for c in asyncflow.submitted if c["fn"] == "generation_task"]
    assert len(gen_calls) == flow.flow_config.num_generators
    for i, call in enumerate(gen_calls):
        assert call["kwargs"]["_input_key"] == f"generator_{i}_input"
        assert call["kwargs"]["_output_key"] == f"generator_{i}_output"
        assert call["kwargs"]["batch_size"] == 4

    # --- Scorer tasks were spawned, one per (scorer, reward_func) ---------
    scorer_calls = [c for c in asyncflow.submitted if c["fn"] == "scorer_task"]
    assert len(scorer_calls) == (
        flow.flow_config.num_scorers * len(trainer.reward_funcs)
    )
    for call in scorer_calls:
        assert call["kwargs"]["reward_func"] is fake_reward
        assert "fake_reward" in call["kwargs"]["_input_key"]
        assert "fake_reward" in call["kwargs"]["_output_key"]

    # --- Learner drove one iteration with preset values -------------------
    assert len(flow.rl.iterations_yielded) == 1
    state = flow.rl.iterations_yielded[0]
    assert state.iteration == 0
    assert state.metric_value == PRESET_EVAL

    # --- Trainer.train was invoked exactly once with the shared ddict -----
    assert len(trainer.train_calls) == 1
    assert isinstance(trainer.train_calls[0]["workflow_ddict"], FakeDDict)
    assert trainer.train_calls[0]["model_config"].base_model_name == "fake/model"


@pytest.mark.asyncio
async def test_full_mock_workflow_propagates_evaluate_value(patched_runtime):
    """The scalar returned by evaluate_func surfaces as the iteration metric."""
    from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig

    @Workflow.evaluate_task
    async def evaluate(model_config):
        return 42.0

    flow = SequentialFlow(
        model_config=ModelConfig(base_model_name="fake/model"),
        trainer=PlaceholderTrainer(reward_funcs=[]),
        evaluate_func=evaluate,
        asyncflow=FakeAsyncflow(),
        flow_config=SequentialFlowConfig(num_generators=1, num_scorers=1),
    )

    await flow.launch()

    assert flow.rl.iterations_yielded[0].metric_value == 42.0
