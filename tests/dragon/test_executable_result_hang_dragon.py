"""An executable task finishes, its side effect is on disk, its future hangs.

    dragon -s tests/dragon/test_executable_result_hang_dragon.py

A minimal, self-contained reproducer for an upstream report: no ROME-A, just
``dragon`` + ``rhapsody`` (``DragonExecutionBackendV3``) + ``radical.asyncflow``.
It shows that while a long-lived **service** task is running, an ordinary
**executable** task submitted afterwards runs to completion — the shell command
executes and writes its output file — yet the ``asyncio.Future`` asyncflow
returned for it never resolves. ``await`` on that future blocks forever.

An executable task makes the defect unambiguous: the task is a plain shell
command (``sh -c 'echo … > file'``). The file on disk is proof the process ran
and exited 0; a future that is still PENDING long after is proof the completion
was never delivered back to the driver.

The suspected mechanism (see ``test_result_delivery_dragon.py`` for the probe):
rhapsody's monitor sweeps outstanding tasks in order and reads each one's result
key; a still-running service task's key read *blocks* rather than raising, so
every result behind it — including the finished executable task's — is never
delivered.

Measured on a 4-CPU single node:

    scenario        task ran (file)   future resolved
    no service      True              True
    idle service    True              False    <- ran, never resolved

Assumes the executable and the driver share a filesystem (true on one node); the
task writes its marker to a tempdir the driver then stats. Exit status is
non-zero if any task ran without its future resolving. Follow a Dragon run with
``dragon-cleanup-deprecated``.
"""

import asyncio
import os
import shlex
import sys
import tempfile
import time

import dragon  # noqa: F401


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


async def main():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import DragonExecutionBackendV3

    backend = await DragonExecutionBackendV3({"results_ddict_mem": 256 * 1024 ** 2})
    flow = await WorkflowEngine.create(backend=backend)
    d = DDict(managers_per_node=1, n_nodes=1, total_mem=256 * 1024 ** 2)
    ser = d.serialize()

    # An executable task: the body returns the shell command asyncflow runs, and
    # calling the decorated task returns a future for its completion.
    @flow.executable_task
    async def shell(command, task_description={}):  # noqa: B006 - asyncflow reads this default
        return command

    # A service task that only sleeps — enough to trigger the defect.
    @flow.function_task(service=True)
    async def idle_service(serialized, marker, task_description={}):  # noqa: B006
        from dragon.data.ddict import DDict

        DDict.attach(serialized)[marker] = "up"
        for _ in range(6000):
            await asyncio.sleep(0.1)

    workdir = tempfile.mkdtemp(prefix="exec_hang_")
    wait_for = float(os.environ.get("RESOLVE_TIMEOUT", 45))

    async def service_up(marker, secs=30):
        for _ in range(int(secs / 0.5)):
            try:
                d[marker]
                return True
            except Exception:
                await asyncio.sleep(0.5)
        return False

    async def file_appears(path, secs=5):
        for _ in range(int(secs / 0.25)):
            if os.path.isfile(path):
                return True
            await asyncio.sleep(0.25)
        return os.path.isfile(path)

    async def probe(label, name):
        """Submit an executable task that writes a file; report ran vs resolved."""
        out = os.path.join(workdir, f"{name}.out")
        # A plain shell command that provably completes and leaves a file.
        cmd = f"/bin/sh -c {shlex.quote(f'echo ran > {shlex.quote(out)}')}"
        fut = shell(cmd, task_description={})
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=wait_for)
            resolved = True
        except asyncio.TimeoutError:
            resolved = False
        except Exception as ex:
            say(f"   {label}: future raised {type(ex).__name__}: {ex}")
            resolved = True
        # The file is written the instant the command exits; if it is absent
        # even now, the command genuinely did not run.
        ran = await file_appears(out)
        say(f"   {label}: ran(file on disk)={ran} future_resolved={resolved}")
        return ran, resolved

    results = {}

    say("1. executable task, nothing else running")
    results["no service"] = await probe("no service", "w1")

    say("2. start an IDLE service (sleeps only), then the same executable task")
    idle_service(ser, "svc_idle", task_description={})
    say(f"   idle service up: {await service_up('svc_idle')}")
    results["idle service"] = await probe("idle service", "w2")

    say("")
    say(f"{'scenario':<16}{'ran(file)':<12}{'resolved':<10}")
    for name, (r, s) in results.items():
        say(f"{name:<16}{str(r):<12}{str(s):<10}")
    say("")
    broken = [n for n, (r, s) in results.items() if r and not s]
    if broken:
        say(f"RAN BUT FUTURE NEVER RESOLVED: {broken}")
        say("The executable command completed (its output file is on disk) but "
            "asyncflow's future for it stayed PENDING — the completion was never "
            "delivered back to the driver.")
    else:
        say("every executable task that ran also resolved its future")

    d.destroy()
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
