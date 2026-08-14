"""The dummy components driven through the real managers.

Same wiring as ``examples/agnostic/dummy_loop.py``, with the sleeps shortened.
The example is otherwise unverifiable: it needs a Dragon runtime and a workflow
engine to run, so this is what keeps it honest.
"""

import asyncio
import json
import os
import re

import rome
from rome.dummy import CHECKPOINT_FILE, DummyTrainer, dummy_infer, dummy_load, dummy_reward

UUID_OUTPUT = re.compile(r"^model example output \[[0-9a-f-]{36}\]$")


async def _settle(predicate, timeout=5.0, interval=0.01):
    for _ in range(int(timeout / interval)):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


def _manager(asyncflow, tmp_path, *, min_samples=4, train_seconds=0.0, num_streams=1):
    return rome.Manager(
        asyncflow,
        data_config=rome.DataConfig(min_samples=min_samples),
        trainer_config=rome.TrainerConfig(
            trainer=DummyTrainer(train_seconds=train_seconds, gpus=1),
            checkpoint_dir=str(tmp_path / "ckpt"),
            poll_interval=0.01,
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer",
                kind=rome.StreamKind.INFERENCE,
                load_func=dummy_load,
                process_func=dummy_infer,
                num_streams=num_streams,
                batch_size=4,
                poll_interval=0.01,
            )
        ],
    )


def test_stream_serves_placeholder_outputs(asyncflow, tmp_path):
    manager = _manager(asyncflow, tmp_path)

    async def scenario():
        await manager.start()
        ids = manager.stream.submit_batch([f"p{i}" for i in range(4)])
        records = [await manager.stream.get_output(i, timeout=5.0) for i in ids]
        await manager.stop()
        return records

    records = asyncio.run(scenario())
    assert len(records) == 4
    assert all(UUID_OUTPUT.match(r["result"]) for r in records)
    # Every output is distinct, so a stalled stream cannot pass by echoing.
    assert len({r["result"] for r in records}) == 4


def test_the_full_dummy_loop_closes(asyncflow, tmp_path):
    """Serve, score, train, and observe the running stream pick up the round."""
    manager = _manager(asyncflow, tmp_path, min_samples=4)

    async def scenario():
        await manager.start()

        # Round one: the stream is untrained.
        before = []
        for rid in manager.stream.submit_batch([f"p{i}" for i in range(4)]):
            record = await manager.stream.get_output(rid, timeout=5.0)
            before.append(record)
            manager.add_training_data(completion=record["result"], score=1.0)

        # Four scored outputs crosses min_samples, so a round fires on its own.
        await _settle(lambda: manager.model_version == 1)

        # Round two: the same stream, now serving from the published checkpoint.
        after = await manager.stream.get_output(
            manager.stream.submit("p-after"), timeout=5.0
        )
        await manager.stop()
        return before, after

    before, after = asyncio.run(scenario())

    assert all(r["model_version"] == 0 for r in before)
    assert after["model_version"] == 1
    assert UUID_OUTPUT.match(after["result"])

    trainer = manager.trainer.train_task
    assert len(trainer.rounds) == 1
    assert trainer.rounds[0]["samples"] == 4


def test_the_reloaded_model_reads_the_checkpoint_from_disk(asyncflow, tmp_path):
    """Version 1 comes off the filesystem, not from a variable in this process."""
    manager = _manager(asyncflow, tmp_path, min_samples=2)

    async def scenario():
        await manager.start()
        for i in range(2):
            manager.add_training_data(completion=f"c{i}", score=1.0)
        await _settle(lambda: manager.model_version == 1)
        record = await manager.stream.get_output(
            manager.stream.submit("p"), timeout=5.0
        )
        await manager.stop()
        return record

    record = asyncio.run(scenario())
    checkpoint = manager.get_current_model()
    with open(os.path.join(checkpoint, CHECKPOINT_FILE)) as fd:
        assert json.load(fd)["version"] == 1
    assert record["model_version"] == 1


def test_streams_keep_serving_while_a_round_is_in_flight(asyncflow, tmp_path):
    """A slow round must not stall inference --- that is the whole design claim."""
    manager = _manager(asyncflow, tmp_path, min_samples=2, train_seconds=0.4)

    async def scenario():
        await manager.start()
        for i in range(2):
            manager.add_training_data(completion=f"c{i}", score=1.0)

        # Wait until the round is genuinely running, then keep asking.
        await _settle(
            lambda: manager.get_training_status() is rome.TrainerStatus.RUNNING
        )
        served = []
        for _ in range(5):
            record = await manager.stream.get_output(
                manager.stream.submit("mid-round"), timeout=5.0
            )
            served.append(record)

        await _settle(lambda: manager.model_version == 1)
        await manager.stop()
        return served

    served = asyncio.run(scenario())
    assert len(served) == 5
    assert all(UUID_OUTPUT.match(r["result"]) for r in served)


def test_replicas_share_the_batch(asyncflow, tmp_path):
    manager = _manager(asyncflow, tmp_path, num_streams=3)

    async def scenario():
        await manager.start()
        ids = manager.stream.submit_batch([f"p{i}" for i in range(12)])
        records = [await manager.stream.get_output(i, timeout=5.0) for i in ids]
        await manager.stop()
        return records

    records = asyncio.run(scenario())
    assert len(records) == 12
    # Work reached more than one replica, and no output was produced twice.
    assert len({r["stream_index"] for r in records}) > 1
    assert len({r["result"] for r in records}) == 12


def test_dummy_reward_stream_fills_the_corpus_on_its_own(asyncflow, tmp_path):
    """Inference and reward both dummy: no scoring code in the workflow at all."""
    manager = rome.Manager(
        asyncflow,
        data_config=rome.DataConfig(min_samples=3),
        trainer_config=rome.TrainerConfig(
            trainer=DummyTrainer(train_seconds=0.0),
            checkpoint_dir=str(tmp_path / "ckpt"),
            poll_interval=0.01,
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer", load_func=dummy_load, process_func=dummy_infer,
                batch_size=4, poll_interval=0.01,
            ),
            rome.StreamConfig(
                name="score", kind=rome.StreamKind.REWARD,
                process_func=dummy_reward, batch_size=4, poll_interval=0.01,
            ),
        ],
    )

    async def scenario():
        await manager.start()
        for rid in manager.stream.submit_batch(["p0", "p1", "p2"], stream="infer"):
            record = await manager.stream.get_output(rid, stream="infer", timeout=5.0)
            manager.stream.submit(record["result"], stream="score")
        await _settle(lambda: manager.model_version == 1)
        await manager.stop()

    asyncio.run(scenario())
    assert manager.data.total_count == 3
    assert all(
        UUID_OUTPUT.match(r["completion"]) for r in manager.data.get_records()
    )
