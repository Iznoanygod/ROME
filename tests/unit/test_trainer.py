import asyncio

import pytest

from rome.trainer import Trainer


def test_trainer_stores_args():
    def reward(*a, **kw):
        return [0.0]

    t = Trainer(gpus=4, reward_funcs=[reward])
    assert t._gpus == 4
    assert t._reward_funcs == [reward]


def test_trainer_reward_funcs_property():
    def reward(*a, **kw):
        return [0.0]

    t = Trainer(gpus=1, reward_funcs=[reward])
    assert t.reward_funcs == [reward]
    assert t.reward_funcs is t._reward_funcs


def test_trainer_reward_funcs_none_becomes_empty_list():
    t = Trainer(gpus=1, reward_funcs=None)
    assert t.reward_funcs == []


def test_trainer_dataset_defaults_to_none():
    t = Trainer(gpus=1, reward_funcs=[])
    assert t.dataset is None


def test_trainer_dataset_stored_from_ctor():
    sentinel = object()
    t = Trainer(gpus=1, reward_funcs=[], dataset=sentinel)
    assert t.dataset is sentinel


def test_trainer_train_raises_not_implemented():
    """train() is async now; await it and expect NotImplementedError."""
    t = Trainer(gpus=1, reward_funcs=[])
    with pytest.raises(NotImplementedError):
        asyncio.run(t.train(model_config=None))