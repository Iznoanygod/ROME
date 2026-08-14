"""The dummy trainer and inference stream shipped for smoke tests."""

import json
import os
import re

import pytest

from rome.dummy import (
    CHECKPOINT_FILE,
    DEFAULT_TEMPLATE,
    DummyModel,
    DummyTrainer,
    dummy_infer,
    dummy_load,
    dummy_reward,
    read_dummy_checkpoint,
    write_dummy_checkpoint,
)

UUID_OUTPUT = re.compile(
    r"^model example output \[[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
    r"-[0-9a-f]{4}-[0-9a-f]{12}\]$"
)


class FakeContext:
    """Stands in for the StreamContext a real stream task would pass."""

    def __init__(self, model=None):
        self.model = model


# -- trainer ---------------------------------------------------------------

def test_trainer_writes_a_checkpoint_recording_the_round(tmp_path):
    trainer = DummyTrainer(train_seconds=0.0)
    out = trainer.train([{"a": 1}, {"a": 2}], str(tmp_path), model_version=3)

    assert out == str(tmp_path)
    with open(os.path.join(out, CHECKPOINT_FILE)) as fd:
        checkpoint = json.load(fd)
    assert checkpoint["version"] == 3
    assert checkpoint["samples"] == 2


def test_trainer_records_each_round(tmp_path):
    trainer = DummyTrainer(train_seconds=0.0)
    trainer.train([1, 2, 3], str(tmp_path / "v1"), model_version=1)
    trainer.train([1, 2], str(tmp_path / "v2"), model_version=2)

    assert [r["version"] for r in trainer.rounds] == [1, 2]
    assert [r["samples"] for r in trainer.rounds] == [3, 2]


def test_trainer_actually_spends_the_time(tmp_path):
    trainer = DummyTrainer(train_seconds=0.2)
    trainer.train([1], str(tmp_path), model_version=1)
    assert trainer.rounds[0]["seconds"] >= 0.2


def test_trainer_infers_the_version_when_not_given(tmp_path):
    trainer = DummyTrainer(train_seconds=0.0)
    trainer.train([1], str(tmp_path / "a"))
    trainer.train([1], str(tmp_path / "b"))
    assert [r["version"] for r in trainer.rounds] == [1, 2]


def test_trainer_declares_its_resources():
    trainer = DummyTrainer(gpus=4, nodes=2, name="smoke")
    assert (trainer.gpus, trainer.nodes, trainer.name) == (4, 2, "smoke")


def test_fail_every_raises_on_schedule(tmp_path):
    trainer = DummyTrainer(train_seconds=0.0, fail_every=2)
    trainer.train([1], str(tmp_path / "v1"), model_version=1)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        trainer.train([1], str(tmp_path / "v2"), model_version=2)
    assert len(trainer.rounds) == 1     # the failed round left no trace


# -- checkpoints -----------------------------------------------------------

def test_checkpoint_round_trips(tmp_path):
    write_dummy_checkpoint(str(tmp_path), version=7, samples=12)
    assert read_dummy_checkpoint(str(tmp_path))["version"] == 7


def test_reading_a_missing_checkpoint_reports_untrained(tmp_path):
    assert read_dummy_checkpoint(None)["version"] == 0
    assert read_dummy_checkpoint(str(tmp_path / "nope"))["version"] == 0


def test_reading_a_corrupt_checkpoint_reports_untrained(tmp_path):
    (tmp_path / CHECKPOINT_FILE).write_text("not json")
    assert read_dummy_checkpoint(str(tmp_path))["version"] == 0


# -- model -----------------------------------------------------------------

def test_output_has_the_requested_shape():
    assert UUID_OUTPUT.match(DummyModel().generate())


def test_every_output_is_unique():
    model = DummyModel()
    outputs = {model.generate() for _ in range(200)}
    assert len(outputs) == 200


def test_model_reports_the_version_it_was_loaded_from(tmp_path):
    write_dummy_checkpoint(str(tmp_path), version=5, samples=40)
    model = DummyModel(str(tmp_path))
    assert model.version == 5
    assert model.trained_on == 40


def test_an_unloaded_model_is_version_zero():
    assert DummyModel().version == 0


def test_template_can_expose_the_version(tmp_path):
    write_dummy_checkpoint(str(tmp_path), version=2, samples=1)
    model = DummyModel(str(tmp_path), template="v{version} output [{uuid}]")
    assert model.generate().startswith("v2 output [")


def test_generate_batch_matches_the_prompt_count():
    outputs = DummyModel().generate_batch(["a", "b", "c"])
    assert len(outputs) == 3
    assert all(UUID_OUTPUT.match(o) for o in outputs)


# -- stream functions ------------------------------------------------------

def test_dummy_load_reads_the_checkpoint(tmp_path):
    write_dummy_checkpoint(str(tmp_path), version=4, samples=8)
    model = dummy_load(str(tmp_path), FakeContext())
    assert model.version == 4


def test_dummy_load_forwards_load_kwargs(tmp_path):
    model = dummy_load(None, FakeContext(), template="x [{uuid}]", latency=0.01)
    assert model.template == "x [{uuid}]"
    assert model.latency == 0.01


def test_dummy_infer_uses_the_streams_model(tmp_path):
    write_dummy_checkpoint(str(tmp_path), version=6, samples=1)
    ctx = FakeContext(DummyModel(str(tmp_path), template="v{version} [{uuid}]"))
    outputs = dummy_infer(["p1", "p2"], ctx)
    assert len(outputs) == 2
    assert all(o.startswith("v6 [") for o in outputs)


def test_dummy_infer_works_before_any_checkpoint_exists():
    """A stream starts before round one; it must still serve."""
    outputs = dummy_infer(["p"], FakeContext(model=None))
    assert UUID_OUTPUT.match(outputs[0])


def test_dummy_reward_produces_corpus_records():
    records = dummy_reward(["out-a", "out-b"], score=0.5)
    assert records == [
        {"completion": "out-a", "score": 0.5},
        {"completion": "out-b", "score": 0.5},
    ]


def test_default_template_is_the_documented_one():
    assert DEFAULT_TEMPLATE == "model example output [{uuid}]"
