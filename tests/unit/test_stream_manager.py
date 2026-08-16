"""Stream manager: persistent inference/reward tasks and weight hot-swapping."""

import asyncio

import pytest

from rome.stream import Stream, StreamConfig, StreamKind, StreamStatus
from rome.utils import MODEL_PATH_KEY, MODEL_VERSION_KEY


async def _settle(predicate, timeout=2.0, interval=0.01):
    """Wait for a background stream task to reach some state."""
    for _ in range(int(timeout / interval)):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.fixture
def stream(namespace, asyncflow):
    return Stream(namespace, asyncflow)


# -- the managed loop ------------------------------------------------------

def test_requests_are_processed_and_results_come_back(stream):
    config = StreamConfig(
        name="infer",
        process_func=lambda prompts: [p.upper() for p in prompts],
        batch_size=4,
        poll_interval=0.01,
    )

    async def scenario():
        await stream.start(config)
        stream.submit_batch(["a", "b", "c"])
        await _settle(lambda: len(stream.get_outputs(consume=False)) == 3)
        outputs = stream.get_outputs()
        await stream.stop()
        return outputs

    outputs = asyncio.run(scenario())
    assert {o["result"] for o in outputs} == {"A", "B", "C"}
    assert all(o["kind"] == "inference" for o in outputs)


def test_get_output_awaits_one_specific_request(stream):
    config = StreamConfig(
        name="infer", process_func=lambda xs: [x * 2 for x in xs], poll_interval=0.01
    )

    async def scenario():
        await stream.start(config)
        rid = stream.submit(21)
        record = await stream.get_output(rid, timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == 42


def test_get_output_returns_none_on_timeout(stream):
    config = StreamConfig(name="infer", process_func=lambda xs: xs, poll_interval=0.01)

    async def scenario():
        await stream.start(config)
        record = await stream.get_output("never-submitted", timeout=0.05)
        await stream.stop()
        return record

    assert asyncio.run(scenario()) is None


def test_async_process_func_is_supported(stream):
    async def process(xs):
        await asyncio.sleep(0)
        return [x + 1 for x in xs]

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=process, poll_interval=0.01)
        )
        rid = stream.submit(1)
        record = await stream.get_output(rid, timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == 2


def test_process_func_receives_the_context_when_it_asks(stream):
    seen = {}

    def process(xs, ctx):
        seen["index"] = ctx.index
        seen["version"] = ctx.model_version
        return xs

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=process, poll_interval=0.01)
        )
        rid = stream.submit("x")
        await stream.get_output(rid, timeout=2.0)
        await stream.stop()

    asyncio.run(scenario())
    assert seen == {"index": 0, "version": 0}


def test_one_bad_request_does_not_kill_the_stream(stream):
    def process(xs):
        if xs[0] == "bad":
            raise ValueError("nope")
        return [x.upper() for x in xs]

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=process, batch_size=1,
                         poll_interval=0.01)
        )
        bad = stream.submit("bad")
        good = stream.submit("good")
        bad_record = await stream.get_output(bad, timeout=2.0)
        good_record = await stream.get_output(good, timeout=2.0)
        statuses = stream.get_status()
        await stream.stop()
        return bad_record, good_record, statuses

    bad_record, good_record, statuses = asyncio.run(scenario())
    assert "nope" in bad_record["result"]["error"]
    assert good_record["result"] == "GOOD"
    assert statuses == [StreamStatus.RUNNING]


# -- replicas --------------------------------------------------------------

def test_requests_round_robin_across_replicas(stream):
    config = StreamConfig(
        name="infer", num_streams=3, process_func=lambda xs: xs, poll_interval=0.01
    )

    async def scenario():
        tasks = await stream.start(config)
        # Submit before the loops get a chance to drain, so the distribution
        # is observable in the queues.
        ids = [stream.submit(i) for i in range(6)]
        pending = [t.pending for t in tasks]
        await _settle(lambda: len(stream.get_outputs(consume=False)) == 6)
        await stream.stop()
        return ids, pending

    ids, _pending = asyncio.run(scenario())
    assert len(set(ids)) == 6


def test_a_request_is_processed_exactly_once(stream):
    """Replicas claim by popping, so no two tasks see the same request."""
    seen = []

    def process(xs):
        seen.extend(xs)
        return xs

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", num_streams=4, process_func=process,
                         poll_interval=0.01)
        )
        stream.submit_batch(list(range(20)))
        await _settle(lambda: len(stream.get_outputs(consume=False)) == 20)
        await stream.stop()

    asyncio.run(scenario())
    assert sorted(seen) == list(range(20))


# -- weight hot-swapping ---------------------------------------------------

def test_stream_loads_the_published_checkpoint_at_startup(stream, namespace):
    loads = []
    namespace[MODEL_PATH_KEY] = "/ckpt/v0"

    async def scenario():
        await stream.start(
            StreamConfig(
                name="infer",
                load_func=lambda path: loads.append(path) or f"model@{path}",
                process_func=lambda xs, ctx: [ctx.model] * len(xs),
                poll_interval=0.01,
            )
        )
        rid = stream.submit("x")
        record = await stream.get_output(rid, timeout=2.0)
        await stream.stop()
        return record

    record = asyncio.run(scenario())
    assert loads == ["/ckpt/v0"]
    assert record["result"] == "model@/ckpt/v0"


def test_a_new_checkpoint_is_picked_up_between_batches(stream, namespace):
    loads = []

    async def scenario():
        await stream.start(
            StreamConfig(
                name="infer",
                model_path="/ckpt/v0",
                load_func=lambda path: loads.append(path) or f"model@{path}",
                process_func=lambda xs, ctx: [ctx.model] * len(xs),
                poll_interval=0.01,
            )
        )
        first = await stream.get_output(stream.submit("x"), timeout=2.0)

        # This is what the training manager does when a round completes.
        namespace[MODEL_PATH_KEY] = "/ckpt/v1"
        namespace[MODEL_VERSION_KEY] = 1

        second = await stream.get_output(stream.submit("y"), timeout=2.0)
        await stream.stop()
        return first, second

    first, second = asyncio.run(scenario())
    assert first["result"] == "model@/ckpt/v0"
    assert second["result"] == "model@/ckpt/v1"
    assert second["model_version"] == 1
    assert loads == ["/ckpt/v0", "/ckpt/v1"]


def test_auto_reload_off_keeps_the_startup_weights(stream, namespace):
    async def scenario():
        await stream.start(
            StreamConfig(
                name="infer",
                model_path="/ckpt/v0",
                auto_reload=False,
                load_func=lambda path: f"model@{path}",
                process_func=lambda xs, ctx: [ctx.model] * len(xs),
                poll_interval=0.01,
            )
        )
        namespace[MODEL_PATH_KEY] = "/ckpt/v1"
        namespace[MODEL_VERSION_KEY] = 1
        record = await stream.get_output(stream.submit("x"), timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == "model@/ckpt/v0"


def test_explicit_reload_model_waits_for_the_swap(stream):
    async def scenario():
        await stream.start(
            StreamConfig(
                name="infer",
                model_path="/ckpt/v0",
                auto_reload=False,
                load_func=lambda path: f"model@{path}",
                process_func=lambda xs, ctx: [ctx.model] * len(xs),
                poll_interval=0.01,
            )
        )
        await stream.reload_model("/ckpt/manual", wait_for_reload=True, timeout=2.0)
        record = await stream.get_output(stream.submit("x"), timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == "model@/ckpt/manual"


def test_on_checkpoint_hook_triggers_a_reload(stream):
    async def scenario():
        await stream.start(
            StreamConfig(
                name="infer",
                model_path="/ckpt/v0",
                auto_reload=False,
                load_func=lambda path: f"model@{path}",
                process_func=lambda xs, ctx: [ctx.model] * len(xs),
                poll_interval=0.01,
            )
        )
        # Exactly what Manager registers on the training manager.
        stream.on_checkpoint("/ckpt/v7", 7)
        record = await stream.get_output(stream.submit("x"), timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == "model@/ckpt/v7"


# -- submission / lifecycle ------------------------------------------------

def test_streams_are_submitted_to_asyncflow_as_services(stream, asyncflow):
    config = StreamConfig(
        name="infer", num_streams=2, num_gpus=2, process_func=lambda xs: xs,
        poll_interval=0.01,
    )

    async def scenario():
        await stream.start(config)
        await stream.stop()

    asyncio.run(scenario())
    assert len(asyncflow.submissions) == 2
    assert all(s["service"] for s in asyncflow.submissions)
    assert all(s["task_description"] == {"gpus_per_rank": 2}
               for s in asyncflow.submissions)


def test_drain_on_stop_finishes_queued_requests(stream):
    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=lambda xs: [x * 2 for x in xs],
                         batch_size=2, poll_interval=0.05)
        )
        stream.submit_batch([1, 2, 3, 4])
        await stream.stop(timeout=2.0)
        return stream.get_outputs()

    outputs = asyncio.run(scenario())
    assert sorted(o["result"] for o in outputs) == [2, 4, 6, 8]


def test_a_self_driven_stream_func_gets_the_same_context(stream):
    async def my_loop(ctx):
        while not ctx.should_stop():
            ctx.maybe_reload()
            for rid, payload in ctx.next_requests():
                ctx.emit(rid, f"custom:{payload}")
            await asyncio.sleep(0.01)

    async def scenario():
        await stream.start(StreamConfig(name="infer", stream_func=my_loop))
        record = await stream.get_output(stream.submit("x"), timeout=2.0)
        await stream.stop()
        return record

    assert asyncio.run(scenario())["result"] == "custom:x"


def test_two_groups_run_side_by_side(stream):
    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=lambda xs: [f"gen:{x}" for x in xs],
                         poll_interval=0.01)
        )
        await stream.start(
            StreamConfig(name="score", kind=StreamKind.REWARD,
                         process_func=lambda xs: [len(x) for x in xs],
                         poll_interval=0.01)
        )
        gen = await stream.get_output(stream.submit("a", stream="infer"),
                                      stream="infer", timeout=2.0)
        score = await stream.get_output(stream.submit("abc", stream="score"),
                                        stream="score", timeout=2.0)
        report = stream.report()
        await stream.stop()
        return gen, score, report

    gen, score, report = asyncio.run(scenario())
    assert gen["result"] == "gen:a"
    assert score["result"] == 3
    assert report["score"]["kind"] == "reward"


def test_ambiguous_stream_name_is_rejected(stream):
    async def scenario():
        await stream.start(StreamConfig(name="a", process_func=lambda xs: xs))
        await stream.start(StreamConfig(name="b", process_func=lambda xs: xs))
        with pytest.raises(ValueError, match="several streams"):
            stream.submit("x")
        await stream.stop()

    asyncio.run(scenario())


def test_config_without_any_work_function_is_rejected():
    with pytest.raises(ValueError, match="process_func"):
        StreamConfig(name="infer").validate()


def test_on_output_hook_sees_every_result(stream):
    seen = []

    async def scenario():
        await stream.start(
            StreamConfig(name="score", kind=StreamKind.REWARD,
                         process_func=lambda xs: [len(x) for x in xs],
                         on_output=seen.append, poll_interval=0.01)
        )
        stream.submit_batch(["a", "bb"])
        await _settle(lambda: len(seen) == 2)
        await stream.stop()

    asyncio.run(scenario())
    assert sorted(r["result"] for r in seen) == [1, 2]


# -- per-group dictionaries ------------------------------------------------

def test_each_group_gets_its_own_dictionary(namespace, asyncflow):
    """Scan cost per poll must not grow with the corpus, so groups are split."""
    stream = Stream(namespace, asyncflow)

    async def scenario():
        await stream.start(
            StreamConfig(name="a", process_func=lambda xs: xs, poll_interval=0.01)
        )
        await stream.start(
            StreamConfig(name="b", process_func=lambda xs: xs, poll_interval=0.01)
        )
        stream.submit("to-a", stream="a")
        backing_a = stream._namespace("a").ddict
        backing_b = stream._namespace("b").ddict
        await stream.stop()
        return backing_a, backing_b

    backing_a, backing_b = asyncio.run(scenario())
    assert backing_a is not backing_b


def test_stream_traffic_stays_out_of_the_managers_dictionary(ddict, asyncflow):
    """The manager's dictionary holds the checkpoint, not per-request keys."""
    from rome.utils import Namespace

    root = Namespace(ddict, "rome|")
    stream = Stream(root, asyncflow)

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=lambda xs: xs, poll_interval=0.01)
        )
        ids = stream.submit_batch(list(range(20)))
        await _settle(lambda: len(stream.get_outputs(consume=False)) == 20)
        await stream.stop()
        return ids

    asyncio.run(scenario())
    # Nothing the streams did touched the manager's dictionary at all.
    assert ddict == {}


def test_a_supplied_group_dictionary_is_not_destroyed(namespace, asyncflow):
    """A dictionary the workflow owns outlives the stream that borrowed it."""
    stream = Stream(namespace, asyncflow)
    mine = {}

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", ddict=mine,
                         process_func=lambda xs: xs, poll_interval=0.01)
        )
        record = await stream.get_output(stream.submit("x"), timeout=2.0)
        await stream.stop()
        stream.close()
        return record

    assert asyncio.run(scenario())["result"] == "x"
    # close() released nothing it did not allocate; `mine` is still usable.
    mine["still-here"] = True
    assert mine["still-here"] is True


def test_close_releases_allocated_dictionaries(namespace, asyncflow):
    stream = Stream(namespace, asyncflow)

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=lambda xs: xs, poll_interval=0.01)
        )
        backing = stream._namespace("infer").ddict
        await stream.stop()
        assert stream._owned, "the manager should own the dictionary it allocated"
        stream.close()
        stream.close()          # idempotent
        return backing

    backing = asyncio.run(scenario())
    assert not stream._owned
    assert backing == {}        # destroy() cleared it


def test_results_survive_stop_until_close(namespace, asyncflow):
    """drain_on_stop is pointless if stopping also throws the results away."""
    stream = Stream(namespace, asyncflow)

    async def scenario():
        await stream.start(
            StreamConfig(name="infer", process_func=lambda xs: [x * 2 for x in xs],
                         batch_size=2, poll_interval=0.05)
        )
        stream.submit_batch([1, 2, 3, 4])
        await stream.stop(timeout=2.0)
        drained = stream.get_outputs()
        stream.close()
        return drained

    assert sorted(o["result"] for o in asyncio.run(scenario())) == [2, 4, 6, 8]
