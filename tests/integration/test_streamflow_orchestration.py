"""Mocked integration tests for rome.flows.streamflow.StreamFlow.

Mirrors the SequentialFlow orchestration tests: drives each scheduler /
gatherer coroutine through a single iteration with a one-shot event, then
asserts on the resulting shared ddict.

The pattern under test (cf. protein_generation/proteinstream.py) is:

    generators ── produce completions per prompt ──> generator_outputs
        |                                                    |
        |                                                    v
    schedulers/gatherers fan out (prompt, idx) work to scorers
        |                                                    |
        v                                                    v
    scorers ── produce per-completion rewards ───> reward_outputs
                                                             |
                                                             v
                                                     trainer rollout

The end-to-end test stitches one tick of every coroutine together so a
fake generator's output flows all the way through to merged rewards —
the same shape proteinstream.py's pipeline produces, minus the model.
"""
from __future__ import annotations

import asyncio

import pytest

from rome.config import ModelConfig
from rome.flows.streamflow import StreamFlow, StreamFlowConfig
from rome.trainer import Trainer


class OneShotEvent:
    """``is_set()`` returns False the first ``n`` calls, then True."""

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


def test_prompt_schedule_mirrors_pool_into_every_generator_queue():
    flow = _make_flow(num_generators=3)

    ddict = {
        "generation_prompts": ["p1", "p2", "p3"],
        "generator_0_input": {},
        "generator_1_input": {},
        "generator_2_input": {},
    }

    asyncio.run(flow._prompt_schedule(ddict, OneShotEvent()))

    expected = {"p1": "p1", "p2": "p2", "p3": "p3"}
    assert ddict["generator_0_input"] == expected
    assert ddict["generator_1_input"] == expected
    assert ddict["generator_2_input"] == expected


def test_prompt_schedule_is_idempotent_on_existing_entries():
    """Re-running the scheduler with a partially-filled queue must not
    duplicate prompts already present — the streaming pool is meant to
    be continuously mirrored, not appended.
    """
    flow = _make_flow(num_generators=1)
    ddict = {
        "generation_prompts": ["p1", "p2"],
        "generator_0_input": {"p1": "p1"},
    }

    asyncio.run(flow._prompt_schedule(ddict, OneShotEvent(n_iterations=2)))

    assert ddict["generator_0_input"] == {"p1": "p1", "p2": "p2"}


def test_generation_gather_appends_new_completions_per_prompt():
    flow = _make_flow(num_generators=2)

    comp_a = {"prompt_ids": [1], "completion_ids": [11], "logprobs": [0.1]}
    comp_b = {"prompt_ids": [1], "completion_ids": [12], "logprobs": [0.2]}
    comp_c = {"prompt_ids": [2], "completion_ids": [21], "logprobs": [0.3]}

    ddict = {
        "generator_outputs": {},
        "generator_0_output": {"p1": [comp_a]},
        "generator_1_output": {"p1": [comp_b], "p2": [comp_c]},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent()))

    # Per-prompt buckets contain completions from every generator
    assert ddict["generator_outputs"]["p1"] == [comp_a, comp_b]
    assert ddict["generator_outputs"]["p2"] == [comp_c]


def test_generation_gather_does_not_double_promote_within_one_tick():
    """Two ticks against the same producer state must not re-append
    completions the gatherer has already promoted.
    """
    flow = _make_flow(num_generators=1)

    comp = {"completion_ids": [1]}
    ddict = {
        "generator_outputs": {},
        "generator_0_output": {"p1": [comp]},
    }

    asyncio.run(flow._generation_gather(ddict, OneShotEvent(n_iterations=2)))

    assert ddict["generator_outputs"]["p1"] == [comp]


def test_scorer_schedule_fans_each_completion_to_one_scorer_per_reward():
    def reward_a(*a, **kw):
        return [0.0]
    def reward_b(*a, **kw):
        return [0.0]

    flow = _make_flow(
        reward_funcs=[reward_a, reward_b],
        num_scorers=2,
    )

    comp_p1_0 = {"completion_ids": [1]}
    comp_p1_1 = {"completion_ids": [2]}
    comp_p2_0 = {"completion_ids": [3]}

    ddict = {
        "generator_outputs": {
            "p1": [comp_p1_0, comp_p1_1],
            "p2": [comp_p2_0],
        },
        "reward_reward_a_0_input": {},
        "reward_reward_a_1_input": {},
        "reward_reward_b_0_input": {},
        "reward_reward_b_1_input": {},
    }

    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent()))

    expected_keys = {("p1", 0), ("p1", 1), ("p2", 0)}
    for fn_name in ("reward_a", "reward_b"):
        q0 = ddict[f"reward_{fn_name}_0_input"]
        q1 = ddict[f"reward_{fn_name}_1_input"]
        # Each (prompt, idx) key lands in exactly one of the scorer queues
        # for this reward func — no duplication, no drops.
        assert set(q0) | set(q1) == expected_keys
        assert set(q0).isdisjoint(set(q1))


def test_scorer_schedule_skips_already_submitted_completions():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=1)

    ddict = {
        "generator_outputs": {"p1": [{"completion_ids": [1]}]},
        "reward_reward_a_0_input": {},
    }

    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent(n_iterations=2)))

    # Single entry on the queue regardless of how many ticks ran.
    assert list(ddict["reward_reward_a_0_input"].keys()) == [("p1", 0)]


def test_scorer_gather_merges_per_scorer_outputs():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=2)

    ddict = {
        "reward_reward_a_outputs": {},
        "reward_reward_a_0_output": {("p1", 0): 0.42},
        "reward_reward_a_1_output": {("p2", 0): 0.99},
    }

    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    assert ddict["reward_reward_a_outputs"] == {
        ("p1", 0): 0.42,
        ("p2", 0): 0.99,
    }


def test_scorer_gather_does_not_overwrite_existing_scores():
    def reward_a(*a, **kw):
        return [0.0]

    flow = _make_flow(reward_funcs=[reward_a], num_scorers=1)

    ddict = {
        # Already merged from a previous round — newer round should not
        # clobber the existing value.
        "reward_reward_a_outputs": {("p1", 0): 0.5},
        "reward_reward_a_0_output": {("p1", 0): 0.1},
    }

    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    assert ddict["reward_reward_a_outputs"] == {("p1", 0): 0.5}


def test_basic_stream_generation_pipeline_end_to_end():
    """End-to-end one-tick run of every scheduler/gatherer against a single
    ddict — proves the four coroutines compose into a working streaming
    pipeline.

    Simulates the steady state of proteinstream.py at a tick boundary:

      1. Prompts are mirrored to every generator's input queue.
      2. Generators (faked here) have produced completions per prompt.
      3. Gather promotes those into the shared ``generator_outputs`` pool.
      4. Scorer schedule fans (prompt, idx) work out to scorer queues.
      5. Scorers (faked here) have produced rewards per (prompt, idx).
      6. Scorer gather merges all rewards into ``reward_*_outputs``.

    After one tick of every stage, every completion the generators
    produced has an attached reward in the merged output — the
    invariant the trainer's rollout polls for.
    """
    def reward_top_prob(*a, **kw):
        return [0.0]

    flow = _make_flow(
        reward_funcs=[reward_top_prob],
        num_generators=2,
        num_scorers=2,
        prompts=["fam_a", "fam_b"],
    )

    comp_a_g0 = {"prompt_ids": [10], "completion_ids": [100], "logprobs": [0.1]}
    comp_a_g1 = {"prompt_ids": [10], "completion_ids": [101], "logprobs": [0.2]}
    comp_b_g0 = {"prompt_ids": [20], "completion_ids": [200], "logprobs": [0.3]}

    ddict = {
        # stage 1 input
        "generation_prompts": ["fam_a", "fam_b"],
        "generator_0_input": {},
        "generator_1_input": {},
        # stage 2 input — fake completions already produced
        "generator_0_output": {"fam_a": [comp_a_g0], "fam_b": [comp_b_g0]},
        "generator_1_output": {"fam_a": [comp_a_g1]},
        # stage 3 sink
        "generator_outputs": {},
        # stage 4 input/sink
        "reward_reward_top_prob_0_input": {},
        "reward_reward_top_prob_1_input": {},
        # stage 5 sinks — will be populated by the fake scorer below
        "reward_reward_top_prob_0_output": {},
        "reward_reward_top_prob_1_output": {},
        # stage 6 sink
        "reward_reward_top_prob_outputs": {},
    }

    # Stage 1: prompt pool → per-generator queues.
    asyncio.run(flow._prompt_schedule(ddict, OneShotEvent()))
    assert set(ddict["generator_0_input"]) == {"fam_a", "fam_b"}
    assert set(ddict["generator_1_input"]) == {"fam_a", "fam_b"}

    # Stage 2: per-generator outputs → shared bucket.
    asyncio.run(flow._generation_gather(ddict, OneShotEvent()))
    assert ddict["generator_outputs"]["fam_a"] == [comp_a_g0, comp_a_g1]
    assert ddict["generator_outputs"]["fam_b"] == [comp_b_g0]

    # Stage 3: shared bucket → scorer input queues.
    asyncio.run(flow._scorer_schedule(ddict, OneShotEvent()))
    routed_keys = (
        set(ddict["reward_reward_top_prob_0_input"])
        | set(ddict["reward_reward_top_prob_1_input"])
    )
    assert routed_keys == {("fam_a", 0), ("fam_a", 1), ("fam_b", 0)}

    # Stage 4: fake out the scorers — each writes the score that the
    # workflow's real scorer_task would have written.
    def fake_score(completion):
        return float(completion["completion_ids"][0])

    for q_key, out_key in (
        ("reward_reward_top_prob_0_input", "reward_reward_top_prob_0_output"),
        ("reward_reward_top_prob_1_input", "reward_reward_top_prob_1_output"),
    ):
        for key, completion in ddict[q_key].items():
            ddict[out_key][key] = fake_score(completion)

    # Stage 5: per-scorer outputs → merged rewards.
    asyncio.run(flow._scorer_gather(ddict, OneShotEvent()))

    merged = ddict["reward_reward_top_prob_outputs"]
    assert merged == {
        ("fam_a", 0): 100.0,
        ("fam_a", 1): 101.0,
        ("fam_b", 0): 200.0,
    }
