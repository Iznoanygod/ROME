"""Test fixtures and shared helpers.

Stubs out heavyweight third-party modules (transformers, trl, dragon, rose,
radical) when they aren't installed so unit and mocked-integration tests can
run on a minimal environment. Tests that need the real packages should call
``pytest.importorskip`` explicitly.

The Dragon stubs are *working* stand-ins rather than empty classes: ROME-A's
components are written against the DDict mapping protocol and Event's
set/clear/is_set, so a plain ``dict`` and a ``threading.Event`` exercise the
real code paths without a Dragon runtime.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from functools import wraps

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
        import torch
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
        import trl
    except ImportError:
        m = _ensure_module("trl")
        m.GRPOConfig = _StubConfig
        m.SFTConfig = _StubConfig
        m.GRPOTrainer = type("GRPOTrainer", (), {})

    try:
        import peft
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
        # A DDict is a mapping shared across nodes; in-process, a dict is a
        # faithful stand-in for everything ROME-A asks of it.
        ddict.DDict = dict
        data.ddict = ddict
        native = _ensure_module("dragon.native")
        event = _ensure_module("dragon.native.event")
        event.Event = threading.Event
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

    try:
        import datasets  # noqa: F401
    except ImportError:
        m = _ensure_module("datasets")

        class _Dataset(list):
            """Enough of ``datasets.Dataset`` for ROME-A's trainers."""

            @classmethod
            def from_list(cls, rows):
                return cls(rows)

            @property
            def column_names(self):
                return sorted({k for row in self for k in row})

        m.Dataset = _Dataset


_stub_if_missing()


class FakeWorkflowEngine:
    """Minimal stand-in for ``radical.asyncflow.WorkflowEngine``.

    Mirrors the real engine's contract for the one thing ROME-A uses:
    ``function_task`` is usable bare or called with ``service=...``, only
    accepts coroutine functions, pops a ``task_description`` kwarg at call
    time, and returns a future. Submissions are recorded so tests can assert
    on what ROME-A asked the backend for.
    """

    def __init__(self):
        self.submissions = []

    def function_task(self, possible_func=None, service=False, **_):
        def actual_decorator(func):
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Function {func.__name__} must be async.")

            @wraps(func)
            def wrapped(*args, **kwargs):
                description = kwargs.pop("task_description", None) or {}
                self.submissions.append(
                    {
                        "name": func.__name__,
                        "task_description": description,
                        "service": service,
                    }
                )
                return asyncio.ensure_future(func(*args, **kwargs))

            return wrapped

        if callable(possible_func):
            return actual_decorator(possible_func)
        return actual_decorator


@pytest.fixture
def asyncflow():
    """A fake workflow engine ROME-A can submit tasks to."""
    return FakeWorkflowEngine()


@pytest.fixture
def ddict():
    """A shared dictionary standing in for a Dragon DDict."""
    return {}


@pytest.fixture
def namespace(ddict):
    """A ROME-A namespace over the shared dictionary."""
    from rome.utils import Namespace

    return Namespace(ddict, "rome|")
