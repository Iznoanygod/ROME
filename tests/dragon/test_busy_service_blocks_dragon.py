"""A service task doing DDict keys()+pop blocks every later task from dispatch.

    dragon -s tests/dragon/test_busy_service_blocks_dragon.py

No ROME-A here -- plain Dragon, rhapsody and asyncflow. Expected output:

    q1 (no service)      RAN
    q2 (idle service)    RAN
    q3 (busy service)    BLOCKED

An idle service that only sleeps does not block anything. A service that polls
a dictionary the way any claim-by-pop worker does wedges the backend's
dispatch, and nothing is ever scheduled again.

This is what stops a ROME-A training round from running while inference
streams are up: the round is submitted, TrainerStatus goes to RUNNING, and the
task is never dispatched. It is not capacity -- it reproduces with slots free.
"""

import asyncio
import sys
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def forever(ser, marker):
    """Idle service: sleeps only."""
    from dragon.data.ddict import DDict

    DDict.attach(ser)[marker] = "up"
    for _ in range(3000):
        await asyncio.sleep(0.1)


async def forever_busy(ser, marker):
    """Service that polls the dictionary the way a ROME-A stream does:
    a full keys() scan, then a pop of anything it finds."""
    from dragon.data.ddict import DDict

    d = DDict.attach(ser)
    d[marker] = "up"
    for _ in range(3000):
        try:
            for k in list(d.keys()):
                if isinstance(k, str) and k.startswith("work|"):
                    try:
                        d.pop(k)
                    except Exception:
                        pass
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def quick(ser, marker):
    from dragon.data.ddict import DDict

    DDict.attach(ser)[marker] = "ran"
    return marker


async def main():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import DragonExecutionBackendV3

    backend = await DragonExecutionBackendV3({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)
    d = DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2)
    ser = d.serialize()

    def has(k):
        try:
            d[k]
            return True
        except Exception:
            return False

    async def wait(k, secs=25):
        for _ in range(int(secs / 0.5)):
            if has(k):
                return True
            await asyncio.sleep(0.5)
        return False

    say("1. plain function task, nothing else running")
    flow.function_task()(quick)(ser, "q1", task_description={})
    say(f"   -> {'RAN' if await wait('q1') else 'BLOCKED'}")

    say("2. start an IDLE service that never returns")
    flow.function_task(service=True)(forever)(ser, "svc_idle", task_description={})
    say(f"   idle service up: {await wait('svc_idle')}")

    say("3. plain function task, after the idle service")
    flow.function_task()(quick)(ser, "q2", task_description={})
    say(f"   -> {'RAN' if await wait('q2') else 'BLOCKED'}")

    say("4. start a BUSY service -- keys()+pop, like a ROME-A stream")
    flow.function_task(service=True)(forever_busy)(ser, "svc_busy", task_description={})
    say(f"   busy service up: {await wait('svc_busy')}")

    say("5. plain function task, after the busy service")
    flow.function_task()(quick)(ser, "q3", task_description={})
    say(f"   -> {'RAN' if await wait('q3', 40) else 'BLOCKED'}")

    say("")
    say(f"q1 (no service)      {'RAN' if has('q1') else 'BLOCKED'}")
    say(f"q2 (idle service)    {'RAN' if has('q2') else 'BLOCKED'}")
    say(f"q3 (busy service)    {'RAN' if has('q3') else 'BLOCKED'}")

    d.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
