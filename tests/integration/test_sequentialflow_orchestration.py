"""Mocked integration tests for rome.flows.sequentialflow.SequentialFlow.

Exercises the scheduler/gatherer coroutines (the parts that don't require
asyncflow / dragon / a real model). One-shot fake events drive each loop
through exactly one iteration so we can assert on the resulting ddict state.
"""
from __future__ import annotations

import asyncio

import pytest

from rome.config import ModelConfig
from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig
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


def _make_flow(*, reward_funcs=(), num_generators=2, num_scorers=2):
    trainer = Trainer(gpus=1, reward_funcs=list(reward_funcs))
    flow = SequentialFlow(
        model_config=ModelConfig(),
        trainer=trainer,
        evaluate_func=lambda mc: 0.0,
        asyncflow=object(),
        flow_config=SequentialFlowConfig(
            num_generators=num_generators, num_scorers=num_scorers,
        ),
    )
    return flow


def test_construction_wires_rl_and_flow_config():
    flow = _make_flow()
    assert flow.rl is not None
    assert flow.flow_config.num_generators == 2
    assert flow.flow_config.num_scorers == 2
    assert flow._generator_tasks == []
    assert flow._scorer_tasks == []


def test_generation_schedule_balances_requests_across_generators():
    flow = _make_flow(num_generators=2)

    ddict = {
        "generation_requests": {"r1": "p1", "r2": "p2", "r3": "p3"},
        "generator_0_input": {},
        "generator_1_input": {},
    }

    asyncio.run(flow._generation_schedule(ddict, OneShotEvent()))

    in0 = ddict["generator_0_input"]
    in1 = ddict["generator_1_input"]
    # All three requests are routed
    assert set(in0) | set(in1) == {"r1", "r2", "r3"}
    # Routing balances: difference in queue sizes <= 1
    assert abs(len(in0) - len(in1)) <= 1
    # And the prompt payloads survive intact
    for rid, payload in {**in0, **in1}.items():
        assert ddict["generation_requests"][rid] == payload


def test_generation_gather_promotes_per_generator_outputs():
    flow = _make_flow(num_generators=2)

    ddict = {
        "generator_outputs": {},
        "generator_0_output": {"r1": {"completion_ids": [1]}},
        "generator_1_output": {"r2": {"completion_ids": [2]}},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent()))

    assert ddict["generator_outputs"] == {
        "r1": {"completion_ids": [1]},
        "r2": {"completion_ids": [2]},
    }


def test_generation_gather_does_not_overwrite_existing():
    flow = _make_flow(num_generators=1)

    sentinel = {"completion_ids": [999]}
    ddict = {
        "generator_outputs": {"r1": sentinel},
        "generator_0_output": {"r1": {"completion_ids": [1]}},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent()))

    assert ddict["generator_outputs"]["r1"] is sentinel


def test_scorer_schedule_fans_out_to_every_reward_func():
    def reward_a(*a, **kw):
        return [0.0]
    def reward_b(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a, reward_b], num_scorers=2)

    ddict = {
        "generator_outputs": {
            "r1": {"completion_ids": [1]},
            "r2": {"completion_ids": [2]},
        },
        "reward_reward_a_0_input": {},
        "reward_reward_a_1_input": {},
        "reward_reward_b_0_input": {},
        "reward_reward_b_1_input": {},
    }

    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent()))

    # Every request lands in some scorer queue for every reward func
    for fn_name in ("reward_a", "reward_b"):
        merged = {
            **ddict[f"reward_{fn_name}_0_input"],
            **ddict[f"reward_{fn_name}_1_input"],
        }
        assert set(merged) == {"r1", "r2"}


def test_scorer_gather_aggregates_per_scorer_outputs():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=2)

    ddict = {
        "reward_reward_a_outputs": {},
        "reward_reward_a_0_output": {"r1": 0.5},
        "reward_reward_a_1_output": {"r2": 0.7},
    }

    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    assert ddict["reward_reward_a_outputs"] == {"r1": 0.5, "r2": 0.7}