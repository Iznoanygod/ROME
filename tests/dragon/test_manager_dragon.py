"""Dragon smoke test: the whole ROME-A loop on a real DDict.

    dragon -s tests/dragon/test_manager_dragon.py

Not a pytest module — the Dragon launcher runs a script, not a test session, so
this is a plain script that exits non-zero on failure. The pytest suite covers
the same logic against a plain dict; what only Dragon can answer is whether the
concurrency assumptions hold against a real distributed dictionary:

* many threads hitting one dictionary (each needs its own client handle, which
  :func:`rome.utils.thread_handle` provides — sharing one corrupts its channel);
* ``pop`` as an exactly-once claim under contention;
* records surviving concurrent writers;
* a prefix scan running while another thread deletes the keys it is streaming.

Always follow a Dragon run with ``dragon-cleanup-deprecated``.
"""

import os
import sys

# `dragon -s` puts this script's directory on sys.path, and the Dragon
# execution backend re-runs the script from a different working directory, so
# neither cwd nor sys.path[0] can be relied on to find the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"ok    {name}")


async def settle(predicate, timeout=60.0, interval=0.05):
    for _ in range(int(timeout / interval)):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def run():
    from dragon.data.ddict import DDict
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import DragonExecutionBackendV3

    import rome
    from rome.dummy import DummyTrainer, dummy_infer, dummy_load

    backend = await DragonExecutionBackendV3(
        {"results_ddict_mem": int(os.environ.get("ROME_RESULTS_MEM", 512 * 1024 ** 2))}
    )
    flow = await WorkflowEngine.create(backend=backend)

    # Shared with a "host workflow" key, to prove ROME-A stays in its namespace.
    ddict = DDict(managers_per_node=1, n_nodes=1,
                  total_mem=int(os.environ.get("ROME_DDICT_MEM", 256 * 1024 ** 2)))
    ddict["host_workflow_state"] = {"campaign": "smoke"}

    checkpoint_dir = tempfile.mkdtemp(prefix="rome_dragon_")
    trainer = DummyTrainer(train_seconds=0.5, gpus=1)

    manager = rome.Manager(
        flow,
        ddict=ddict,
        data_config=rome.DataConfig(min_samples=8),
        trainer_config=rome.TrainerConfig(
            trainer=trainer, checkpoint_dir=checkpoint_dir, poll_interval=0.2
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer",
                load_func=dummy_load,
                process_func=dummy_infer,
                num_streams=4,
                batch_size=2,
                poll_interval=0.02,
            )
        ],
    )

    try:
        await manager.start()

        # -- streams under contention: 4 replicas, one shared dictionary ----
        request_ids = manager.stream.submit_batch([f"p{i}" for i in range(40)])
        got = await settle(
            lambda: len(manager.stream.get_outputs(consume=False)) == 40, timeout=90
        )
        records = manager.stream.get_outputs()

        def every_request_answered_exactly_once():
            assert got, f"only {len(records)}/40 results arrived"
            assert len(records) == 40, len(records)
            returned = [r["request_id"] for r in records]
            assert len(set(returned)) == 40, "a request was answered twice"
            assert set(returned) == set(request_ids), "result ids do not match requests"

        def work_reached_several_replicas():
            replicas = {r["stream_index"] for r in records}
            assert len(replicas) > 1, f"only replica {replicas} did any work"

        def outputs_are_distinct():
            results = [r["result"] for r in records]
            assert len(set(results)) == 40, "a stream echoed a cached result"

        check("every request answered exactly once", every_request_answered_exactly_once)
        check("work spread over replicas", work_reached_several_replicas)
        check("outputs are distinct", outputs_are_distinct)

        # -- concurrent writers into the corpus -----------------------------
        def concurrent_adds_all_survive():
            """Threads adding at once is the case key-per-record exists for."""
            before = manager.data.total_count
            errors = []

            def writer(tag):
                try:
                    for i in range(25):
                        manager.add_training_data(completion=f"{tag}-{i}", score=1.0)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    traceback.print_exc()

            threads = [threading.Thread(target=writer, args=(f"w{t}",)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, errors[0]
            added = manager.data.total_count - before
            assert added == 100, f"{added}/100 records survived concurrent writers"

        check("concurrent writers lose nothing", concurrent_adds_all_survive)

        # -- the closed loop -------------------------------------------------
        reached = await settle(lambda: manager.model_version >= 1, timeout=90)

        def training_fired_and_published():
            assert reached, "no training round completed"
            assert manager.get_current_model(), "no checkpoint was published"
            assert trainer.rounds, "the trainer task never ran"

        check("training fired and published", training_fired_and_published)

        version = manager.model_version
        after = await manager.stream.get_output(
            manager.stream.submit("after-training"), timeout=60
        )

        def streams_swapped_onto_the_new_checkpoint():
            assert after is not None, "stream stopped answering after a round"
            assert after["model_version"] >= version, (
                f"stream still on v{after['model_version']}, published v{version}"
            )

        check("streams swapped onto the checkpoint", streams_swapped_onto_the_new_checkpoint)

        # -- namespacing -----------------------------------------------------
        def host_workflow_keys_untouched():
            assert ddict["host_workflow_state"] == {"campaign": "smoke"}
            stray = [
                k for k in ddict.keys()
                if isinstance(k, str) and not k.startswith("rome|")
                and k != "host_workflow_state"
            ]
            assert not stray, f"ROME-A wrote outside its namespace: {stray[:5]}"

        check("host workflow keys untouched", host_workflow_keys_untouched)

        await manager.stop()
    finally:
        try:
            await flow.shutdown()
        except Exception:
            traceback.print_exc()
        ddict.destroy()


def main():
    try:
        asyncio.run(run())
    except Exception:
        traceback.print_exc()
        FAILURES.append("run() raised")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("ROME-A works on Dragon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
