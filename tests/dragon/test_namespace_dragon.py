"""Dragon smoke test: does ROME-A's DDict layout actually work on a real DDict?

Run under the Dragon runtime, which the rest of the test suite cannot do::

    dragon -s tests/dragon/test_namespace_dragon.py

This is not a pytest module — the Dragon launcher runs a script, not a test
session. It exits non-zero on the first failure.

What it checks is exactly the set of DDict operations :mod:`rome.utils` relies
on, and that is a deliberately short list: single-key get/set/delete, a
``keys()`` scan, and picklable values. Everything above it in ROME-A is built
from those, so if they hold here the rest follows.
"""

import sys
import traceback

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


def main():
    from dragon.data.ddict import DDict
    from dragon.native.event import Event

    from rome.utils import MODEL_VERSION_KEY, Namespace

    ddict = DDict(managers_per_node=1, n_nodes=1, total_mem=(1024 ** 3))

    try:
        ns = Namespace(ddict, "rome|")

        def single_key_roundtrip():
            ns["a"] = 1
            assert ns["a"] == 1, ns["a"]

        def missing_key_returns_default():
            # rome.utils.Namespace.get catches (KeyError, TypeError); if Dragon
            # raises something else, every `.get(..., default)` in ROME-A breaks.
            assert ns.get("nope") is None
            assert ns.get("nope", 7) == 7

        def records_survive_the_round_trip():
            record = {"uid": "d1", "score": 91.5, "seq": "MKV", "nested": [1, 2]}
            ns.namespace("record")["d1"] = record
            assert ns.namespace("record")["d1"] == record

        def prefix_scan_finds_only_its_namespace():
            records = ns.namespace("record")
            for i in range(5):
                records[f"r{i}"] = {"i": i}
            ns.namespace("meta")["consumed"] = 3
            ddict["someone_elses_key"] = "host workflow state"

            keys = records.keys()
            assert sorted(keys) == ["d1", "r0", "r1", "r2", "r3", "r4"], keys
            assert ns.namespace("meta").keys() == ["consumed"]

        def delete_and_pop():
            records = ns.namespace("record")
            assert records.pop("r0") == {"i": 0}
            assert records.pop("r0", "gone") == "gone"
            assert "r0" not in records.keys()

        def drain_is_a_claim_protocol():
            queue = ns.namespace("queue")
            for i in range(10):
                queue[f"q{i}"] = i
            first = Namespace(ddict, "rome|").namespace("queue").drain(limit=4)
            second = Namespace(ddict, "rome|").namespace("queue").drain()
            keys_a = {k for k, _ in first}
            keys_b = {k for k, _ in second}
            assert len(keys_a) == 4, keys_a
            assert not (keys_a & keys_b), keys_a & keys_b
            assert len(keys_a | keys_b) == 10

        def increment_counter():
            meta = ns.namespace("meta")
            assert meta.increment("count") == 1
            assert meta.increment("count", 4) == 5

        def model_version_defaults_to_zero():
            fresh = Namespace(ddict, "unused|")
            assert int(fresh.get(MODEL_VERSION_KEY, 0)) == 0

        def host_workflow_keys_are_untouched():
            # ROME-A namespaces everything so a shared DDict is safe.
            assert ddict["someone_elses_key"] == "host workflow state"

        def event_contract():
            event = Event()
            assert not event.is_set()
            event.set()
            assert event.is_set()
            event.clear()
            assert not event.is_set()

        check("single-key round trip", single_key_roundtrip)
        check("missing key returns default", missing_key_returns_default)
        check("dict records survive pickling", records_survive_the_round_trip)
        check("prefix scan is namespace-scoped", prefix_scan_finds_only_its_namespace)
        check("delete and pop", delete_and_pop)
        check("drain claims exactly once", drain_is_a_claim_protocol)
        check("increment counter", increment_counter)
        check("model version defaults to 0", model_version_defaults_to_zero)
        check("host workflow keys untouched", host_workflow_keys_are_untouched)
        check("Event set/clear/is_set", event_contract)
    finally:
        ddict.destroy()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all DDict/Event checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
