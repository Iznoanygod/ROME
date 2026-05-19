"""Mocked integration tests for rome.train.GRPO.

These don't load real models or call trl, they verify the orchestration:
reward-function wrapping (`_reward_func_wrapper`) and default rollout
(`_default_rollout_func`) correctly read/write the shared workflow_ddict.
"""
from __future__ import annotations

import asyncio
import itertools
import uuid

import pytest

from rome.train import GRPO


def _make_grpo(reward_funcs):
    return GRPO(gpus=1, dataset=None, reward_funcs=reward_funcs)


def test_grpo_default_config_built_when_none():
    g = _make_grpo(reward_funcs=[])
    assert g._grpo_config is not None
    # Stubbed _StubConfig in conftest stores kwargs as attributes
    assert getattr(g._grpo_config, "learning_rate", None) == 5e-6
    assert getattr(g._grpo_config, "num_generations", None) == 4


def test_grpo_custom_config_preserved():
    sentinel = object()
    g = GRPO(gpus=1, dataset=None, reward_funcs=[], grpo_config=sentinel)
    assert g._grpo_config is sentinel


def test_grpo_stores_callbacks_and_rollout():
    cb = object()
    rf = object()
    g = GRPO(
        gpus=1,
        dataset=None,
        reward_funcs=[],
        trainer_callbacks=[cb],
        rollout_func=rf,
    )
    assert g._trainer_callbacks == [cb]
    assert g._rollout_func is rf


def test_reward_func_wrapper_returns_rewards_in_order():
    def my_reward(*a, **kw):
        return [0.0]

    g = _make_grpo(reward_funcs=[my_reward])
    g._workflow_ddict = {
        "reward_my_reward_outputs": {
            "req-a": 0.1,
            "req-b": 0.9,
            "req-c": 0.5,
        }
    }
    wrapped = g._reward_func_wrapper(my_reward)
    rewards = asyncio.run(
        wrapped(
            prompts=[], completions=[], ground_truths=[],
            request_ids=["req-b", "req-a", "req-c"],
        )
    )
    assert rewards == [0.9, 0.1, 0.5]


def test_reward_func_wrapper_polls_until_outputs_present(monkeypatch):
    """If outputs aren't there yet, the wrapper sleeps and re-reads the ddict."""
    def my_reward(*a, **kw):
        return [0.0]

    g = _make_grpo(reward_funcs=[my_reward])

    # First read: empty. Second read (after a "sleep"): populated.
    states = iter([{}, {"req-1": 0.7}])

    class ProbeDDict:
        def __getitem__(self, key):
            assert key == "reward_my_reward_outputs"
            return next(states)

    g._workflow_ddict = ProbeDDict()

    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("rome.train.grpo.asyncio.sleep", fake_sleep)

    wrapped = g._reward_func_wrapper(my_reward)
    rewards = asyncio.run(
        wrapped(
            prompts=[], completions=[], ground_truths=[],
            request_ids=["req-1"],
        )
    )
    assert rewards == [0.7]
    assert sleep_calls == [1]


def test_default_rollout_func_round_trips_prompts(monkeypatch):
    g = _make_grpo(reward_funcs=[])

    # Pre-fabricate three request ids; each prompt gets the next one.
    ids = ["id-1", "id-2", "id-3"]
    id_iter = iter(ids)
    monkeypatch.setattr(uuid, "uuid4", lambda: next(id_iter))

    # Generator outputs are already there so the inner while-loop exits immediately
    g._workflow_ddict = {
        "generation_requests": {},
        "generator_outputs": {
            "id-1": {"prompt_ids": [1], "completion_ids": [11], "logprobs": [0.1]},
            "id-2": {"prompt_ids": [2], "completion_ids": [22], "logprobs": [0.2]},
            "id-3": {"prompt_ids": [3], "completion_ids": [33], "logprobs": [0.3]},
        },
    }

    out = g._default_rollout_func(prompts=["a", "b", "c"], trainer=None)

    assert out["request_ids"] == ids
    assert out["prompt_ids"] == [[1], [2], [3]]
    assert out["completion_ids"] == [[11], [22], [33]]
    assert out["logprobs"] == [[0.1], [0.2], [0.3]]
    # prompts were written into generation_requests under those ids
    assert g._workflow_ddict["generation_requests"] == {
        "id-1": "a", "id-2": "b", "id-3": "c",
    }


def test_default_rollout_func_waits_for_missing_output(monkeypatch):
    """If generator_outputs is missing a request, rollout sleeps and re-reads."""
    g = _make_grpo(reward_funcs=[])

    monkeypatch.setattr(uuid, "uuid4", lambda: "only-id")

    # Two reads of generator_outputs: first empty, second populated
    reads = iter([
        {},
        {"only-id": {"prompt_ids": [9], "completion_ids": [99], "logprobs": [0.9]}},
    ])
    gen_requests = {}

    class ProbeDDict:
        def __getitem__(self, key):
            if key == "generation_requests":
                return gen_requests
            if key == "generator_outputs":
                return next(reads)
            raise KeyError(key)

        def __setitem__(self, key, value):
            pass

    g._workflow_ddict = ProbeDDict()

    slept = []
    monkeypatch.setattr("rome.train.grpo.time.sleep", lambda s: slept.append(s))

    out = g._default_rollout_func(prompts=["p"], trainer=None)
    assert out["request_ids"] == ["only-id"]
    assert out["completion_ids"] == [[99]]
    assert slept == [1]