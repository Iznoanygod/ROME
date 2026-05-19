"""Test fixtures and shared helpers.

Stubs out heavyweight third-party modules (transformers, trl, dragon, rose,
radical) when they aren't installed so unit and mocked-integration tests can
run on a minimal environment. Tests that need the real packages should call
``pytest.importorskip`` explicitly.
"""
from __future__ import annotations

import sys
import types

import pytest


def pytest_collection_modifyitems(config, items):
    """Tag every test that isn't explicitly `slow` as `fast`.

    Lets callers pick:
        pytest                 # run everything
        pytest -m fast         # unit + smoke + mocked-integration
        pytest -m slow         # the heavyweight end-to-end test
    """
    for item in items:
        if "slow" not in item.keywords:
            item.add_marker(pytest.mark.fast)


class _StubConfig:
    """Drop-in stand-in for trl.GRPOConfig / trl.SFTConfig.

    Stores whatever kwargs callers pass so tests can assert on them.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _stub_if_missing() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        m = _ensure_module("torch")

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        m.no_grad = _NoGrad

    try:
        import transformers  # noqa: F401
    except ImportError:
        m = _ensure_module("transformers")
        m.GenerationConfig = _StubConfig
        m.AutoTokenizer = type("AutoTokenizer", (), {})
        m.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})

    try:
        import trl  # noqa: F401
    except ImportError:
        m = _ensure_module("trl")
        m.GRPOConfig = _StubConfig
        m.SFTConfig = _StubConfig
        m.GRPOTrainer = type("GRPOTrainer", (), {})

    try:
        import peft  # noqa: F401
    except ImportError:
        m = _ensure_module("peft")
        m.get_peft_model = lambda model, cfg: model
        m.LoraConfig = _StubConfig
        m.PeftModel = type("PeftModel", (), {})

    try:
        import radical.asyncflow  # noqa: F401
    except ImportError:
        radical = _ensure_module("radical")
        af = _ensure_module("radical.asyncflow")
        radical.asyncflow = af
        af.WorkflowEngine = type("WorkflowEngine", (), {})

    try:
        import dragon  # noqa: F401
    except ImportError:
        _ensure_module("dragon")
        data = _ensure_module("dragon.data")
        ddict = _ensure_module("dragon.data.ddict")
        ddict.DDict = type("DDict", (), {})
        data.ddict = ddict
        native = _ensure_module("dragon.native")
        event = _ensure_module("dragon.native.event")
        event.Event = type("Event", (), {})
        native.event = event

    try:
        import rose  # noqa: F401
    except ImportError:
        _ensure_module("rose")
        metrics = _ensure_module("rose.metrics")
        metrics.GREATER_THAN_THRESHOLD = "greater_than_threshold"
        learner = _ensure_module("rose.learner")
        learner.SequentialReinforcementLearner = type(
            "SequentialReinforcementLearner",
            (),
            {"__init__": lambda self, asyncflow=None: None},
        )


_stub_if_missing()
