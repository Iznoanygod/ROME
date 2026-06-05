"""Mocked integration tests for rome.flows.streamflow.StreamFlow.

The streaming workflow uses prompts (not request ids) as keys into the
shared ddict. Each generator continuously appends completions to a
per-prompt list; the trainer drains that list when it rolls out. These
tests drive each coroutine through exactly one loop iteration via a
one-shot fake event, then assert on the resulting ddict shape.
"""
from __future__ import annotations

import asyncio

import pytest

from rome.config import ModelConfig
from rome.flows.streamflow import StreamFlow, StreamFlowConfig
from rome.trainer import Trainer


class OneShotEvent:
    """`is_set()` returns False the first N calls, then True."""

    def __init__(self, n_iterations: int = 1):
        self._calls = 0
        self._limit = n_iterations

    def is_set(self) -> bool:
        result = self._calls >= self._limit
        self._calls += 1
        return result


def _make_flow(*, reward_funcs=(), num_generators=2, num_scorers=2, prompts=()):
    trainer = Trainer(gpus=1, reward_funcs=list(reward_funcs))
    flow = StreamFlow(
        model_config=ModelConfig(),
        trainer=trainer,
        evaluate_func=lambda mc: 0.0,
        asyncflow=object(),
        flow_config=StreamFlowConfig(
            num_generators=num_generators,
            num_scorers=num_scorers,
            prompts=list(prompts),
        ),
    )
    return flow


def test_construction_wires_rl_and_flow_config():
    flow = _make_flow(prompts=["p1", "p2"])
    assert flow.rl is not None
    assert flow.flow_config.num_generators == 2
    assert flow.flow_config.num_scorers == 2
    assert flow.flow_config.prompts == ["p1", "p2"]
    assert flow._generator_tasks == []
    assert flow._scorer_tasks == []


def test_prompt_schedule_mirrors_pool_into_every_generator():
    flow = _make_flow(num_generators=3)

    ddict = {
        "generation_prompts": ["p1", "p2", "p3"],
        "generator_0_input": {},
        "generator_1_input": {},
        "generator_2_input": {},
    }

    asyncio.run(flow._prompt_schedule(ddict, OneShotEvent()))

    for i in range(3):
        assert set(ddict[f"generator_{i}_input"]) == {"p1", "p2", "p3"}


def test_prompt_schedule_is_idempotent_on_repeat():
    flow = _make_flow(num_generators=2)

    ddict = {
        "generation_prompts": ["p1"],
        "generator_0_input": {"p1": "p1"},
        "generator_1_input": {},
    }

    asyncio.run(flow._prompt_schedule(ddict, OneShotEvent()))

    assert ddict["generator_0_input"] == {"p1": "p1"}
    assert ddict["generator_1_input"] == {"p1": "p1"}


def test_generation_gather_appends_new_completions_per_prompt():
    flow = _make_flow(num_generators=2)

    ddict = {
        "generator_outputs": {},
        "generator_0_output": {"p1": [{"completion_ids": [1]}]},
        "generator_1_output": {"p1": [{"completion_ids": [2]}]},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent()))

    out = ddict["generator_outputs"]
    assert set(out) == {"p1"}
    completion_ids = [c["completion_ids"] for c in out["p1"]]
    assert sorted(completion_ids) == [[1], [2]]


def test_generation_gather_skips_already_promoted_items_within_loop():
    """A multi-iteration loop should not re-promote completions."""
    flow = _make_flow(num_generators=1)

    ddict = {
        "generator_outputs": {},
        "generator_0_output": {"p1": [{"completion_ids": [1]}]},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent(n_iterations=3)))

    completion_ids = [c["completion_ids"] for c in ddict["generator_outputs"]["p1"]]
    assert completion_ids == [[1]]


def test_generation_gather_picks_up_new_completions_in_subsequent_loops():
    """If the generator appends a new completion mid-flight, gather promotes it.

    The driver pre-loads two completions into the per-generator output dict
    before the loop runs. The gather promotes both in its first body
    execution; the second body execution sees nothing new.
    """
    flow = _make_flow(num_generators=1)

    ddict = {
        "generator_outputs": {},
        "generator_0_output": {
            "p1": [{"completion_ids": [1]}, {"completion_ids": [2]}],
        },
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent(n_iterations=2)))

    completion_ids = [c["completion_ids"] for c in ddict["generator_outputs"]["p1"]]
    assert completion_ids == [[1], [2]]


def test_scorer_schedule_fans_out_to_every_reward_func():
    def reward_a(*a, **kw):
        return [0.0]

    def reward_b(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a, reward_b], num_scorers=2)

    ddict = {
        "generator_outputs": {
            "p1": [{"completion_ids": [1]}, {"completion_ids": [2]}],
            "p2": [{"completion_ids": [3]}],
        },
        "reward_reward_a_0_input": {},
        "reward_reward_a_1_input": {},
        "reward_reward_b_0_input": {},
        "reward_reward_b_1_input": {},
    }

    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent()))

    expected_keys = {("p1", 0), ("p1", 1), ("p2", 0)}
    for fn_name in ("reward_a", "reward_b"):
        merged = {
            **ddict[f"reward_{fn_name}_0_input"],
            **ddict[f"reward_{fn_name}_1_input"],
        }
        assert set(merged) == expected_keys


def test_scorer_schedule_balances_across_scorers():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=2)

    ddict = {
        "generator_outputs": {
            "p1": [{"completion_ids": [i]} for i in range(4)],
        },
        "reward_reward_a_0_input": {},
        "reward_reward_a_1_input": {},
    }

    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent()))

    q0 = ddict["reward_reward_a_0_input"]
    q1 = ddict["reward_reward_a_1_input"]
    # 4 items, 2 scorers -> difference <= 1
    assert abs(len(q0) - len(q1)) <= 1
    assert set(q0) | set(q1) == {("p1", i) for i in range(4)}


def test_scorer_gather_merges_per_scorer_outputs():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=2)

    ddict = {
        "reward_reward_a_outputs": {},
        "reward_reward_a_0_output": {("p1", 0): 0.5},
        "reward_reward_a_1_output": {("p1", 1): 0.7, ("p2", 0): 0.9},
    }

    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    assert ddict["reward_reward_a_outputs"] == {
        ("p1", 0): 0.5,
        ("p1", 1): 0.7,
        ("p2", 0): 0.9,
    }


def test_scorer_gather_does_not_overwrite_existing():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=1)

    sentinel = object()
    ddict = {
        "reward_reward_a_outputs": {("p1", 0): sentinel},
        "reward_reward_a_0_output": {("p1", 0): "replacement"},
    }

    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    assert ddict["reward_reward_a_outputs"][("p1", 0)] is sentinel
