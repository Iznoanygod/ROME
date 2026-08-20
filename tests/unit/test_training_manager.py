"""Training manager: when a round fires, where it runs, what it publishes."""

import asyncio

import pytest

from rome.data import DataConfig, DataManager
from rome.train.base import FunctionTrainer, TrainTask
from rome.trainer import Trainer, TrainerConfig, TrainerStatus
from rome.utils import MODEL_PATH_KEY, MODEL_VERSION_KEY


class RecordingTrainer(TrainTask):
    """A trainer that records what it was handed instead of training."""

    def __init__(self, **kwargs):
        super().__init__(name="recording", **kwargs)
        self.calls = []

    def train(self, dataset, output_dir, **kwargs):
        self.calls.append({"dataset": list(dataset), "output_dir": output_dir,
                           "kwargs": kwargs})
        return output_dir


@pytest.fixture
def data(namespace):
    return DataManager(namespace, DataConfig(min_samples=2))


@pytest.fixture
def trainer(namespace, data, asyncflow, tmp_path):
    task = RecordingTrainer()
    return Trainer(
        namespace,
        data,
        asyncflow,
        TrainerConfig(
            trainer=task,
            checkpoint_dir=str(tmp_path / "ckpt"),
            auto_train=False,
        ),
    )


# -- status ----------------------------------------------------------------

def test_status_starts_not_started(trainer):
    assert trainer.status is TrainerStatus.NOT_STARTED


def test_status_reports_whether_a_round_is_possible(trainer, data):
    asyncio.run(trainer.start())
    assert trainer.status is TrainerStatus.NOT_ENOUGH_DATA
    data.add(score=1.0)
    data.add(score=2.0)
    assert trainer.status is TrainerStatus.WAITING


def test_train_declines_without_enough_data(trainer, data):
    data.add(score=1.0)
    assert asyncio.run(trainer.train()) is None
    assert trainer.status is TrainerStatus.NOT_ENOUGH_DATA
    assert trainer.train_task.calls == []


def test_force_trains_anyway(trainer, data):
    data.add(score=1.0)
    checkpoint = asyncio.run(trainer.train(force=True))
    assert checkpoint is not None
    assert len(trainer.train_task.calls) == 1


# -- publishing ------------------------------------------------------------

def test_round_publishes_checkpoint_and_bumps_version(trainer, data, namespace):
    data.add(score=1.0)
    data.add(score=2.0)
    checkpoint = asyncio.run(trainer.train())

    assert namespace[MODEL_PATH_KEY] == checkpoint
    assert namespace[MODEL_VERSION_KEY] == 1
    assert trainer.get_current_model() == checkpoint
    assert trainer.rounds_completed == 1


def test_checkpoint_path_carries_the_version(trainer, data):
    for _ in range(2):
        data.add(score=1.0)
    first = asyncio.run(trainer.train())
    for _ in range(2):
        data.add(score=1.0)
    second = asyncio.run(trainer.train())
    assert first.endswith("recording/v1")
    assert second.endswith("recording/v2")


def test_round_consumes_the_corpus(trainer, data):
    for _ in range(2):
        data.add(score=1.0)
    asyncio.run(trainer.train())
    assert not data.ready_to_train()
    assert data.total_count == 2


def test_checkpoint_callbacks_fire_with_path_and_version(trainer, data):
    seen = []
    trainer.on_checkpoint(lambda path, version: seen.append((path, version)))
    for _ in range(2):
        data.add(score=1.0)
    checkpoint = asyncio.run(trainer.train())
    assert seen == [(checkpoint, 1)]


def test_a_broken_callback_does_not_lose_the_checkpoint(trainer, data, namespace):
    trainer.on_checkpoint(lambda path, version: 1 / 0)
    for _ in range(2):
        data.add(score=1.0)
    checkpoint = asyncio.run(trainer.train())
    assert namespace[MODEL_PATH_KEY] == checkpoint


# -- scheduling ------------------------------------------------------------

def test_round_is_submitted_to_asyncflow(trainer, data, asyncflow):
    for _ in range(2):
        data.add(score=1.0)
    asyncio.run(trainer.train())
    assert len(asyncflow.submissions) == 1
    submission = asyncflow.submissions[0]
    assert submission["name"] == "rome_train_recording"
    assert submission["service"] is False


def test_trainer_resources_reach_the_backend(namespace, data, asyncflow, tmp_path):
    task = RecordingTrainer(gpus=4, nodes=2)
    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=task, checkpoint_dir=str(tmp_path), auto_train=False),
    )
    data.add(score=1.0)
    asyncio.run(trainer.train(force=True))
    assert asyncflow.submissions[0]["task_description"] == {
        "ranks": 2, "gpus_per_rank": 4
    }


def test_explicit_task_description_wins(namespace, data, asyncflow, tmp_path):
    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(
            trainer=RecordingTrainer(gpus=4),
            checkpoint_dir=str(tmp_path),
            auto_train=False,
            task_description={"policy": "custom"},
        ),
    )
    data.add(score=1.0)
    asyncio.run(trainer.train(force=True))
    assert asyncflow.submissions[0]["task_description"] == {"policy": "custom"}


def test_train_kwargs_reach_the_task(namespace, data, asyncflow, tmp_path):
    task = RecordingTrainer()
    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=task, checkpoint_dir=str(tmp_path), auto_train=False,
                      train_kwargs={"epochs": 7}),
    )
    data.add(score=1.0)
    asyncio.run(trainer.train(force=True))
    assert task.calls[0]["kwargs"]["epochs"] == 7
    assert task.calls[0]["kwargs"]["model_version"] == 1


# -- auto-train loop -------------------------------------------------------

def test_auto_train_fires_when_the_threshold_is_crossed(namespace, data, asyncflow, tmp_path):
    task = RecordingTrainer()
    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=task, checkpoint_dir=str(tmp_path),
                      auto_train=True, poll_interval=0.01),
    )

    async def scenario():
        await trainer.start()
        await asyncio.sleep(0.05)
        assert task.calls == []          # nothing to train on yet
        data.add(score=1.0)
        data.add(score=2.0)
        for _ in range(100):             # let the poll loop notice
            await asyncio.sleep(0.01)
            if task.calls:
                break
        await trainer.stop(timeout=2.0)

    asyncio.run(scenario())
    assert len(task.calls) == 1
    assert trainer.model_version == 1


def test_max_rounds_finishes_the_trainer(namespace, data, asyncflow, tmp_path):
    task = RecordingTrainer()
    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=task, checkpoint_dir=str(tmp_path),
                      auto_train=True, poll_interval=0.01, max_rounds=1),
    )

    async def scenario():
        await trainer.start()
        data.add(score=1.0)
        data.add(score=2.0)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if trainer.status is TrainerStatus.TRAINING_COMPLETE:
                break
        await trainer.stop(timeout=2.0)

    asyncio.run(scenario())
    assert len(task.calls) == 1
    assert trainer.rounds_completed == 1


def test_auto_train_survives_a_failing_round(namespace, data, asyncflow, tmp_path):
    class Boom(TrainTask):
        def train(self, dataset, output_dir, **kwargs):
            raise RuntimeError("gpu on fire")

    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=Boom(name="boom"), checkpoint_dir=str(tmp_path),
                      auto_train=True, poll_interval=0.01),
    )

    async def scenario():
        await trainer.start()
        data.add(score=1.0)
        data.add(score=2.0)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if trainer.last_error:
                break
        await trainer.stop(timeout=2.0)

    asyncio.run(scenario())
    assert "gpu on fire" in trainer.last_error
    assert trainer.model_version == 0     # nothing was published


def test_stop_on_failure_marks_the_trainer_failed(namespace, data, asyncflow, tmp_path):
    class Boom(TrainTask):
        def train(self, dataset, output_dir, **kwargs):
            raise RuntimeError("gpu on fire")

    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=Boom(name="boom"), checkpoint_dir=str(tmp_path),
                      auto_train=True, poll_interval=0.01, stop_on_failure=True),
    )

    async def scenario():
        await trainer.start()
        data.add(score=1.0)
        data.add(score=2.0)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if trainer.status is TrainerStatus.FAILED:
                break

    asyncio.run(scenario())
    assert trainer.status is TrainerStatus.FAILED


# -- trainer plumbing ------------------------------------------------------

def test_a_bare_function_becomes_a_train_task(namespace, data, asyncflow, tmp_path):
    seen = {}

    def my_finetune(dataset, output_dir, **kwargs):
        seen["n"] = len(dataset)
        return output_dir

    trainer = Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=my_finetune, checkpoint_dir=str(tmp_path),
                      auto_train=False),
    )
    data.add(score=1.0)
    asyncio.run(trainer.train(force=True))
    assert isinstance(trainer.train_task, FunctionTrainer)
    assert seen["n"] == 1


def test_start_without_a_trainer_is_an_error(namespace, data, asyncflow):
    trainer = Trainer(namespace, data, asyncflow, TrainerConfig())
    with pytest.raises(ValueError, match="trainer must be set"):
        asyncio.run(trainer.start())


def test_empty_corpus_is_refused_before_scheduling(trainer, asyncflow):
    with pytest.raises(ValueError, match="empty dataset"):
        asyncio.run(trainer.train(force=True))
    assert asyncflow.submissions == []


def test_report_summarizes_the_run(trainer, data):
    for _ in range(2):
        data.add(score=1.0)
    asyncio.run(trainer.train())
    report = trainer.report()
    assert report["model_version"] == 1
    assert report["rounds_completed"] == 1
    assert report["corpus_size"] == 2
    assert report["last_error"] is None


# -- executable-round completion detection (the Dragon result-delivery gap) ----

def _fallback_trainer(namespace, data, asyncflow, tmp_path, grace):
    return Trainer(
        namespace, data, asyncflow,
        TrainerConfig(trainer=RecordingTrainer(),
                      checkpoint_dir=str(tmp_path / "ckpt"),
                      auto_train=False, result_fallback_seconds=grace),
    )


def test_await_round_completes_when_the_marker_appears_not_the_future(
        namespace, data, asyncflow, tmp_path):
    """An executable round is done when its completion marker lands on disk.

    On Dragon the task's future can stay PENDING forever (a running service
    blocks result delivery), so the round must be detected by the per-round
    marker the wrapper writes — promptly, without waiting on the future.
    """
    from rome.trainer import TRAIN_COMPLETE_MARKER

    trainer = _fallback_trainer(namespace, data, asyncflow, tmp_path, grace=0.2)
    outdir = tmp_path / "round"
    outdir.mkdir()
    marker = str(outdir / TRAIN_COMPLETE_MARKER)

    async def scenario():
        fut = asyncio.get_running_loop().create_future()   # never resolves

        async def drop_marker():
            await asyncio.sleep(0.3)
            with open(marker, "w") as fd:
                fd.write("done")

        writer = asyncio.ensure_future(drop_marker())
        result = await asyncio.wait_for(
            trainer._await_round(fut, marker, authoritative=True), timeout=5)
        await writer
        return result, fut.done()

    result, future_done = asyncio.run(scenario())
    assert result == marker
    assert future_done is False        # completion came from disk, not the future


def test_await_round_does_not_complete_on_a_missing_marker(
        namespace, data, asyncflow, tmp_path):
    """No marker yet + pending future ⇒ still waiting.

    Guards the publish_into_repo trap: the checkpoint is a stable path that
    already exists from the prior round, so only the per-round marker — absent
    here — may signal completion.
    """
    from rome.trainer import TRAIN_COMPLETE_MARKER

    trainer = _fallback_trainer(namespace, data, asyncflow, tmp_path, grace=0.2)
    outdir = tmp_path / "round"
    outdir.mkdir()
    marker = str(outdir / TRAIN_COMPLETE_MARKER)      # never created

    async def scenario():
        fut = asyncio.get_running_loop().create_future()   # never resolves
        # _await_round must keep waiting; wrapping it in a 1s timeout must fire.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                trainer._await_round(fut, marker, authoritative=True), timeout=1.0)

    asyncio.run(scenario())
