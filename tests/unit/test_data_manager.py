"""Data manager: scored outputs in, training dataset out."""

import pytest

from rome.data import DataConfig, DataManager


@pytest.fixture
def data(namespace):
    return DataManager(namespace, DataConfig(min_samples=3))


def test_add_returns_uid_and_stores_fields(data):
    uid = data.add(score=0.9, prompt="2+2?", completion="4")
    assert uid is not None
    (record,) = data.get_records()
    assert record["uid"] == uid
    assert record["prompt"] == "2+2?"
    assert record["score"] == 0.9
    assert "added_at" in record


def test_add_works_for_any_domain(data):
    data.add(score=91.2, sequence="MKV", pdb_path="/x/b1.pdb", pTM=0.9)
    (record,) = data.get_records()
    assert record["sequence"] == "MKV"
    assert record["score"] == 91.2


def test_records_come_back_oldest_first(data):
    uids = [data.add(score=float(i), step=i) for i in range(5)]
    assert [r["uid"] for r in data.get_records()] == uids


def test_min_score_rejects_on_the_way_in(namespace):
    data = DataManager(namespace, DataConfig(min_score=0.5))
    assert data.add(score=0.4) is None
    assert data.add(score=0.6) is not None
    assert data.total_count == 1


def test_filter_func_rejects_on_the_way_in(namespace):
    data = DataManager(
        namespace, DataConfig(filter_func=lambda r: r.get("pTM", 0) >= 0.8)
    )
    assert data.add(score=90.0, pTM=0.5) is None
    assert data.add(score=90.0, pTM=0.9) is not None


def test_dedup_key_drops_repeats(namespace):
    data = DataManager(namespace, DataConfig(dedup_key=lambda r: r["sequence"]))
    assert data.add(sequence="MKV") is not None
    assert data.add(sequence="MKV") is None
    assert data.add(sequence="MKW") is not None
    assert data.total_count == 2


def test_dedup_sees_records_added_by_another_node(namespace):
    """The corpus, not process-local memory, is the authority on duplicates."""
    node_a = DataManager(namespace, DataConfig(dedup_key=lambda r: r["sequence"]))
    node_b = DataManager(namespace, DataConfig(dedup_key=lambda r: r["sequence"]))
    node_a.add(sequence="MKV")
    assert node_b.add(sequence="MKV") is None


def test_max_records_evicts_oldest(namespace):
    data = DataManager(namespace, DataConfig(max_records=3))
    for i in range(5):
        data.add(score=float(i), step=i)
    steps = [r["step"] for r in data.get_records()]
    assert steps == [2, 3, 4]


def test_ready_to_train_tracks_the_threshold(data):
    assert not data.ready_to_train()
    for i in range(3):
        data.add(score=float(i))
    assert data.ready_to_train()


def test_mark_consumed_requires_fresh_data_for_the_next_round(data):
    for i in range(3):
        data.add(score=float(i))
    data.mark_consumed()
    assert not data.ready_to_train()
    assert data.total_count == 3  # the corpus itself is monotonic
    for i in range(3):
        data.add(score=float(i))
    assert data.ready_to_train()


def test_consume_on_train_off_retrains_on_everything(namespace):
    data = DataManager(namespace, DataConfig(min_samples=2, consume_on_train=False))
    data.add(score=1.0)
    data.add(score=2.0)
    data.mark_consumed()
    assert data.ready_to_train()


def test_top_k_sampling_picks_the_best(namespace):
    data = DataManager(namespace, DataConfig(sampling="top_k", shard_size=2))
    for score in (0.1, 0.9, 0.5, 0.7):
        data.add(score=score)
    assert [r["score"] for r in data.get_dataset()] == [0.9, 0.7]


def test_recent_sampling_picks_the_newest(namespace):
    data = DataManager(namespace, DataConfig(sampling="recent", shard_size=2))
    for i in range(4):
        data.add(score=float(i), step=i)
    assert [r["step"] for r in data.get_dataset()] == [2, 3]


def test_custom_sample_func_wins(namespace):
    data = DataManager(
        namespace,
        DataConfig(sample_func=lambda records: [{"custom": len(records)}]),
    )
    data.add(score=1.0)
    assert data.get_dataset() == [{"custom": 1}]


def test_unknown_sampling_strategy_is_rejected(namespace):
    data = DataManager(namespace, DataConfig(sampling="vibes"))
    data.add(score=1.0)
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        data.get_dataset()


def test_records_are_stamped_with_the_model_version(namespace):
    from rome.utils import MODEL_VERSION_KEY

    data = DataManager(namespace)
    data.add(score=1.0)
    namespace[MODEL_VERSION_KEY] = 3
    data.add(score=2.0)
    assert [r["model_version"] for r in data.get_records()] == [0, 3]


def test_add_batch_accepts_plain_dicts(data):
    uids = data.add_batch([{"score": 1.0, "prompt": "a"}, {"score": 2.0, "prompt": "b"}])
    assert len(uids) == 2
    assert {r["prompt"] for r in data.get_records()} == {"a", "b"}


def test_concurrent_producers_do_not_clobber_each_other(namespace):
    """Two tasks adding at once both survive — the point of key-per-record."""
    a = DataManager(namespace)
    b = DataManager(namespace)
    for i in range(50):
        a.add(score=float(i), who="a")
        b.add(score=float(i), who="b")
    assert a.total_count == 100
    assert b.total_count == 100


def test_clear_resets_the_corpus(data):
    for i in range(3):
        data.add(score=float(i))
    assert data.clear() == 3
    assert data.total_count == 0
    assert not data.ready_to_train()
