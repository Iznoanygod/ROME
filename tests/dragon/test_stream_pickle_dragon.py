"""Why does a ROME-A stream not start on the Dragon execution backend?

    dragon -s tests/dragon/test_stream_pickle_dragon.py

The Dragon backend runs a function task in a separate `Process`, so everything
the task body closes over has to survive pickling *and* still work on the other
side. `LocalExecutionBackend` runs the body as a thread in the driver, which
hides any failure here — which is why streams work there and not on Dragon.

This isolates each thing a StreamTask carries and reports which one breaks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import multiprocessing as mp
import traceback

import dragon  # noqa: F401  registers the 'dragon' mp start method

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
    except Exception as ex:
        RESULTS.append((name, False))
        print(f"FAIL  {name}: {type(ex).__name__}: {str(ex)[:200]}")
    else:
        RESULTS.append((name, True))
        print(f"ok    {name}" + (f"  ({detail})" if detail else ""))


def _child_touch_event(event, out):
    """Can a child process actually use an Event it received by pickle?"""
    try:
        event.set()
        out.value = 1
    except Exception:
        out.value = 2


def _child_read_namespace(ns, key, out):
    """Can a child process READ the DDict through a pickled Namespace?"""
    try:
        ns.get(key)
        out.value = 1
    except Exception:
        out.value = 2


def _child_write_namespace(ns, key, value, out):
    """...and write back visibly to the driver?"""
    try:
        ns[key] = value
        out.value = 1
    except Exception:
        out.value = 2


def main():
    import cloudpickle
    from dragon.data.ddict import DDict
    from dragon.native.event import Event

    from rome.dummy import dummy_infer, dummy_load
    from rome.stream import StreamConfig, StreamTask, run_stream_task
    from rome.utils import Namespace

    ddict = DDict(managers_per_node=1, n_nodes=1, total_mem=(256 * 1024 ** 2))
    root = Namespace(ddict, "rome|")

    # -- 1. the pieces, one at a time ---------------------------------------
    check("DDict round-trips", lambda: len(cloudpickle.dumps(ddict)))
    check("Namespace round-trips", lambda: len(cloudpickle.dumps(root)))

    event = Event()
    check("Event pickles", lambda: len(cloudpickle.dumps(event)))

    def event_usable_in_child():
        out = mp.Value("i", 0)
        p = mp.Process(target=_child_touch_event, args=(event, out))
        p.start()
        p.join(timeout=30)
        assert out.value == 1, f"child could not set the Event (code {out.value})"
        assert event.is_set(), "the driver never saw the child's set()"
        return "set() crossed processes"

    check("Event works in a child process", event_usable_in_child)

    # -- 2. the whole StreamTask --------------------------------------------
    config = StreamConfig(
        name="infer", load_func=dummy_load, process_func=dummy_infer,
        num_streams=1, batch_size=2, poll_interval=0.02,
    )
    task = StreamTask(0, config, root.namespace("infer"), root)

    check("StreamTask pickles", lambda: len(cloudpickle.dumps(task)))

    def task_round_trips_intact():
        revived = cloudpickle.loads(cloudpickle.dumps(task))
        assert revived.index == task.index
        assert revived.config.name == task.config.name
        return "index and config survived"

    check("StreamTask round-trips intact", task_round_trips_intact)

    # -- 3. the exact closure submit_task() sends ---------------------------
    async def stream_entry(_task=task):
        await run_stream_task(_task)

    stream_entry.__name__ = "rome_stream_infer_0"

    check("stream_entry pickles", lambda: len(cloudpickle.dumps(stream_entry)))

    # -- 4. can a child actually USE the dictionary it received? -------------
    # This is the line run_stream_task blocks on: task.root.get(MODEL_PATH_KEY).
    root["probe_key"] = "hello"

    def child_can_read():
        out = mp.Value("i", 0)
        p = mp.Process(target=_child_read_namespace, args=(root, "probe_key", out))
        p.start()
        p.join(timeout=45)
        if p.is_alive():
            p.terminate()
            raise AssertionError("child BLOCKED reading the DDict (45s)")
        assert out.value == 1, f"child raised reading the DDict (code {out.value})"
        return "read crossed processes"

    check("child process can read the DDict", child_can_read)

    def child_can_write():
        out = mp.Value("i", 0)
        p = mp.Process(target=_child_write_namespace, args=(root, "from_child", 42, out))
        p.start()
        p.join(timeout=45)
        if p.is_alive():
            p.terminate()
            raise AssertionError("child BLOCKED writing the DDict (45s)")
        assert out.value == 1, f"child raised writing the DDict (code {out.value})"
        assert root.get("from_child") == 42, "driver never saw the child's write"
        return "write visible to driver"

    check("child process can write the DDict", child_can_write)

    print("\n" + "=" * 60)
    failed = [name for name, ok in RESULTS if not ok]
    if failed:
        print("first thing that breaks: " + failed[0])
    else:
        print("every piece crosses a process boundary cleanly")
    print("=" * 60)

    ddict.destroy()
    return 1 if failed else 0


if __name__ == "__main__":
    mp.set_start_method("dragon", force=True)
    raise SystemExit(main())
