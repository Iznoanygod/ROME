import pytest

from rome.trainer import Trainer


def test_trainer_stores_args():
    def reward(*a, **kw):
        return [0.0]

    t = Trainer(gpus=4, dataset="dummy-dataset", reward_funcs=[reward])
    assert t._gpus == 4
    assert t._dataset == "dummy-dataset"
    assert t._reward_funcs == [reward]


def test_trainer_reward_funcs_property():
    def reward(*a, **kw):
        return [0.0]

    t = Trainer(gpus=1, dataset=None, reward_funcs=[reward])
    assert t.reward_funcs == [reward]
    assert t.reward_funcs is t._reward_funcs


def test_trainer_reward_funcs_none_becomes_empty_list():
    t = Trainer(gpus=1, dataset=None, reward_funcs=None)
    assert t.reward_funcs == []


def test_trainer_train_raises_not_implemented():
    t = Trainer(gpus=1, dataset=None, reward_funcs=[])
    with pytest.raises(NotImplementedError):
        t.train(model_config=None)
