import asyncio

import pytest

from rome.workflow import Workflow


def test_workflow_constructor_stores_attrs():
    sentinel_mc = object()
    sentinel_tr = object()
    sentinel_ef = object()
    sentinel_af = object()
    wf = Workflow(
        model_config=sentinel_mc,
        trainer=sentinel_tr,
        evaluate_func=sentinel_ef,
        asyncflow=sentinel_af,
    )
    assert wf.model_config is sentinel_mc
    assert wf.trainer is sentinel_tr
    assert wf.evaluate_func is sentinel_ef
    assert wf.asyncflow is sentinel_af


def test_reward_task_decorator_tags_function():
    @Workflow.reward_task
    def my_reward(prompts, completions, ground_truth, **kwargs):
        return [1.0] * len(completions)

    assert getattr(my_reward, "_is_reward_task", False) is True


def test_evaluate_task_decorator_tags_function():
    @Workflow.evaluate_task
    def my_eval(model_config):
        return 0.42

    assert getattr(my_eval, "_is_evaluate_task", False) is True


def test_workflow_launch_raises_not_implemented():
    wf = Workflow(
        model_config=None,
        trainer=None,
        evaluate_func=None,
        asyncflow=None,
    )
    with pytest.raises(NotImplementedError):
        asyncio.run(wf.launch())
