"""Where does task dispatch stop working on DragonExecutionBackendV3?

    dragon -s tests/dragon/test_backend_dispatch_dragon.py

ROME-A's streams do not start on the multi-process Dragon backend. Everything
smaller has been ruled out -- see docs/dragon.md -- so this bisects the one
remaining step:

  step 1  hand-rolled StreamTask, submitted directly, no Manager
  step 2  ...then construct rome.Manager (do not start it)
  step 3  ...then manager.start(), and submit another by hand

Steps 1 and 2 pass and step 3 fails, which says StreamManager.start() both
fails to dispatch its own task and takes the backend's queue down with it.
Requires `pip install -e .` -- the task process imports rome by name.
"""

import asyncio
import sys
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def main():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import DragonExecutionBackendV3

    import rome
    from rome.dummy import dummy_infer, dummy_load
    from rome.stream import StreamConfig, StreamTask, run_stream_task
    from rome.utils import Namespace

    backend = await DragonExecutionBackendV3({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)
    d = DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2)
    root = Namespace(d, "rome|")

    def cfg(name):
        return StreamConfig(name=name, load_func=dummy_load, process_func=dummy_infer,
                            num_streams=1, batch_size=2, poll_interval=0.05, num_gpus=0)

    async def submit_hand(label, index, group_ns):
        task = StreamTask(index=index, config=cfg("hand"), ddict=group_ns, root=root)

        async def entry(_task=task):
            await run_stream_task(_task)

        entry.__name__ = f"hand_{index}"
        flow.function_task(service=True)(entry)(task_description={})
        for _ in range(30):
            if group_ns.namespace("status").get(str(index)) == "RUNNING":
                say(f"  {label}: RUNNING")
                return True
            await asyncio.sleep(0.5)
        say(f"  {label}: NEVER STARTED (15s) "
            f"status={group_ns.namespace('status').get(str(index))!r}")
        return False

    g1 = Namespace(DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2), "")
    say("STEP 1: hand submit, no Manager anywhere")
    step1 = await submit_hand("step1", 1, g1)

    say("STEP 2: construct rome.Manager (not started)")
    manager = rome.Manager(flow, ddict=d, stream_configs=[cfg("infer")])
    g2 = Namespace(DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2), "")
    step2 = await submit_hand("step2", 2, g2)

    say("STEP 3: manager.start(), then hand submit again")
    await manager.start()
    g3 = Namespace(DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2), "")
    step3 = await submit_hand("step3", 3, g3)

    say("")
    say(f"step1 (no Manager)        -> {'OK' if step1 else 'BROKEN'}")
    say(f"step2 (Manager built)     -> {'OK' if step2 else 'BROKEN'}")
    say(f"step3 (Manager started)   -> {'OK' if step3 else 'BROKEN'}")
    say(f"manager's own stream      -> "
        f"{manager.stream.stream_tasks[0].status if manager.stream.stream_tasks else '?'}")

    d.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
