"""The DDict layout ROME-A's cross-node correctness rests on."""

from rome.utils import Namespace


def test_namespace_prefixes_keys(ddict):
    ns = Namespace(ddict, "rome|")
    ns["a"] = 1
    assert ddict == {"rome|a": 1}
    assert ns["a"] == 1


def test_child_namespaces_share_the_ddict(ddict):
    root = Namespace(ddict, "rome|")
    records = root.namespace("record")
    records["x"] = {"uid": "x"}
    assert ddict["rome|record|x"] == {"uid": "x"}
    assert records.keys() == ["x"]
    # The parent sees the child's keys, prefixed.
    assert root.keys("record|") == ["record|x"]


def test_namespace_accepts_non_string_segments(ddict):
    ns = Namespace(ddict, "").namespace("req", 3)
    ns["r1"] = "payload"
    assert ddict == {"req|3|r1": "payload"}


def test_get_and_pop_defaults(ddict):
    ns = Namespace(ddict, "n|")
    assert ns.get("missing") is None
    assert ns.get("missing", 7) == 7
    assert ns.pop("missing", "fallback") == "fallback"
    ns["k"] = "v"
    assert ns.pop("k") == "v"
    assert "k" not in ns


def test_prefix_scan_ignores_other_namespaces(ddict):
    root = Namespace(ddict, "rome|")
    root.namespace("record")["a"] = 1
    root.namespace("meta")["b"] = 2
    ddict["someone_elses_key"] = 3
    assert root.namespace("record").keys() == ["a"]
    assert root.namespace("meta").values() == [2]
    # The host workflow's own keys are untouched by a ROME-A scan.
    assert ddict["someone_elses_key"] == 3


def test_drain_removes_what_it_returns(ddict):
    ns = Namespace(ddict, "q|")
    for i in range(5):
        ns[f"r{i}"] = i
    first = ns.drain(limit=2)
    assert len(first) == 2
    assert len(ns.keys()) == 3
    rest = ns.drain()
    assert len(rest) == 3
    assert ns.keys() == []


def test_drain_is_a_claim_protocol(ddict):
    """Two consumers draining the same queue never get the same item."""
    ns = Namespace(ddict, "q|")
    for i in range(10):
        ns[f"r{i}"] = i

    consumer_a = Namespace(ddict, "q|").drain(limit=4)
    consumer_b = Namespace(ddict, "q|").drain(limit=10)

    keys_a = {k for k, _ in consumer_a}
    keys_b = {k for k, _ in consumer_b}
    assert not (keys_a & keys_b)
    assert len(keys_a | keys_b) == 10


def test_increment_starts_from_zero(ddict):
    ns = Namespace(ddict, "n|")
    assert ns.increment("count") == 1
    assert ns.increment("count", 4) == 5
    assert ns["count"] == 5
