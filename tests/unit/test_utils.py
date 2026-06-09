"""Unit tests for rome.utils save/reload/version helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rome.config import ModelConfig
from rome.utils import (
    WEIGHT_VERSION_KEY,
    bump_weight_version,
    read_weight_version,
    reload_lora,
    reload_model,
    save_model,
)


# ---------- save_model ----------

def test_save_model_writes_adapter_when_lora_name_set():
    model = MagicMock()
    cfg = ModelConfig(base_model_name="b", lora_name="/tmp/adapter-x")
    path = save_model(model, cfg)
    assert path == "/tmp/adapter-x"
    model.save_pretrained.assert_called_once_with("/tmp/adapter-x")


def test_save_model_writes_full_model_when_only_model_name_set():
    model = MagicMock()
    cfg = ModelConfig(model_name="/tmp/model-x")
    path = save_model(model, cfg)
    assert path == "/tmp/model-x"
    model.save_pretrained.assert_called_once_with("/tmp/model-x")


def test_save_model_prefers_lora_over_model_name():
    model = MagicMock()
    cfg = ModelConfig(model_name="/tmp/model", lora_name="/tmp/lora")
    path = save_model(model, cfg)
    assert path == "/tmp/lora"
    model.save_pretrained.assert_called_once_with("/tmp/lora")


def test_save_model_raises_when_no_target_configured():
    cfg = ModelConfig(base_model_name="base-only")
    with pytest.raises(ValueError):
        save_model(MagicMock(), cfg)


# ---------- reload_lora ----------

def test_reload_lora_requires_lora_name():
    cfg = ModelConfig(model_name="/tmp/m")
    with pytest.raises(ValueError):
        reload_lora(MagicMock(), cfg)


def test_reload_lora_raises_when_path_missing(tmp_path):
    cfg = ModelConfig(lora_name=str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError):
        reload_lora(MagicMock(), cfg)


def test_reload_lora_applies_loaded_state_in_place(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg = ModelConfig(lora_name=str(adapter_dir))
    model = MagicMock()

    fake_state = {"layer.weight": "tensor-sentinel"}
    monkeypatch.setattr("peft.load_peft_weights", lambda path, **kw: fake_state)

    applied = {}
    def _apply(m, sd):
        applied["model"] = m
        applied["state"] = sd
    monkeypatch.setattr("peft.set_peft_model_state_dict", _apply)

    reload_lora(model, cfg)

    assert applied["model"] is model
    assert applied["state"] is fake_state


# ---------- reload_model dispatch ----------

def test_reload_model_dispatches_to_lora_when_lora_name_set(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg = ModelConfig(model_name="/tmp/m", lora_name=str(adapter_dir))

    called = {}
    def _fake_lora(model, model_config):
        called["lora"] = True
    monkeypatch.setattr("rome.utils.reload_lora", _fake_lora)

    reload_model(MagicMock(), cfg)
    assert called == {"lora": True}


def test_reload_model_full_path_loads_state_dict(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cfg = ModelConfig(model_name=str(model_dir))

    src_model = MagicMock()
    src_model.state_dict.return_value = {"w": 1}

    class FakeAuto:
        @staticmethod
        def from_pretrained(path):
            assert path == str(model_dir)
            return src_model

    monkeypatch.setattr("transformers.AutoModelForCausalLM", FakeAuto)

    target = MagicMock()
    reload_model(target, cfg)
    target.load_state_dict.assert_called_once_with({"w": 1})


def test_reload_model_raises_when_nothing_configured():
    cfg = ModelConfig(base_model_name="base-only")
    with pytest.raises(ValueError):
        reload_model(MagicMock(), cfg)


def test_reload_model_raises_when_full_path_missing(tmp_path):
    cfg = ModelConfig(model_name=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        reload_model(MagicMock(), cfg)


# ---------- weight version counter ----------

def test_read_weight_version_default_on_missing():
    assert read_weight_version({}) == 0
    assert read_weight_version({}, default=5) == 5


def test_read_weight_version_returns_stored_value():
    assert read_weight_version({WEIGHT_VERSION_KEY: 7}) == 7


def test_read_weight_version_falls_back_on_unparseable():
    assert read_weight_version({WEIGHT_VERSION_KEY: "junk"}, default=3) == 3


def test_bump_weight_version_increments_from_zero():
    ddict = {}
    assert bump_weight_version(ddict) == 1
    assert ddict[WEIGHT_VERSION_KEY] == 1


def test_bump_weight_version_increments_from_existing():
    ddict = {WEIGHT_VERSION_KEY: 4}
    assert bump_weight_version(ddict) == 5
    assert ddict[WEIGHT_VERSION_KEY] == 5


def test_bump_weight_version_recovers_from_corrupt_value():
    ddict = {WEIGHT_VERSION_KEY: "not-an-int"}
    assert bump_weight_version(ddict) == 1
    assert ddict[WEIGHT_VERSION_KEY] == 1
