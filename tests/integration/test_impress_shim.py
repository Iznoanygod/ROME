"""Tests for the ROME-under-IMPRESS shim.

Mocks the surface of IMPRESS (``ImpressManager``, ``ProteinBindingPipeline``,
the per-pass CSV write) with a fake harness that runs N adaptive cycles
in tmpdir, calling the wrapped ``adaptive_fn`` each pass. Verifies that
ROME's corpus + training trigger are exercised correctly under the
IMPRESS contract — without IMPRESS actually being installed.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytest

from rome.impress import CorpusThresholds, RomeShim
from rome.protein.hooks import TaskHooks


# ---------------------------------------------------------------------------
# Fake IMPRESS — mirrors the subset of the real interface the shim reads.
# ---------------------------------------------------------------------------

@dataclass
class FakeProteinBindingPipeline:
    name: str
    passes: int = 0
    sub_order: int = 0
    seq_rank: int = 0
    current_scores: Dict[str, float] = field(default_factory=dict)
    previous_scores: Dict[str, float] = field(default_factory=dict)
    iter_seqs: Dict[str, Any] = field(default_factory=dict)
    af_out_path: str = "."


class FakeManager:
    """Stand-in for ImpressManager. The shim's ``attached`` only uses it as
    an opaque handle in the current revision; this fake just stores the
    pipeline list so the harness can iterate."""

    def __init__(self):
        self.pipelines: List[FakeProteinBindingPipeline] = []

    async def run_passes(
        self,
        pipeline: FakeProteinBindingPipeline,
        n_passes: int,
        csv_writer: Callable[[FakeProteinBindingPipeline], None],
        adaptive_fn: Callable[[Any], Awaitable[Any]],
    ) -> None:
        """Simulate IMPRESS running its s1..s5 stage chain + adaptive_fn per pass."""
        self.pipelines.append(pipeline)
        for p in range(n_passes):
            pipeline.passes = p
            csv_writer(pipeline)
            await adaptive_fn(pipeline)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as fd:
        fd.write("ID,avg_plddt,ptm,avg_pae\n")
        for r in rows:
            fd.write(f"{r['ID']},{r['plddt']},{r['ptm']},{r['pae']}\n")


# ---------------------------------------------------------------------------
# A dummy trainer hook the tests can inspect.
# ---------------------------------------------------------------------------

@dataclass
class TrainRecorder:
    calls: List[Dict[str, Any]] = field(default_factory=list)


def _train_hook(rec: TrainRecorder):
    async def train(config, sampled_entries, output_checkpoint_dir):
        os.makedirs(output_checkpoint_dir, exist_ok=True)
        with open(os.path.join(output_checkpoint_dir, "weights.pt"), "wb") as fd:
            fd.write(b"\x00")
        rec.calls.append({
            "shard_size": len(sampled_entries),
            "ckpt": output_checkpoint_dir,
        })
        return output_checkpoint_dir
    return train


def _make_shim(tmp_path, train_recorder, **overrides) -> RomeShim:
    return RomeShim(
        corpus_thresholds=CorpusThresholds(min_pLDDT=80.0, min_pTM=0.80, max_pAE=5.0),
        train_batch_threshold=overrides.get("train_batch_threshold", 2),
        train_shard_size=overrides.get("train_shard_size", 16),
        mpnn_checkpoint_dir=str(tmp_path / "ckpts"),
        base_path=str(tmp_path),
        task_hooks=TaskHooks(mpnn_train=_train_hook(train_recorder)),
        csv_path_for=lambda p: str(tmp_path / f"af_stats_{p.name}_pass_{p.passes}.csv"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_wrap_adaptive_fn_calls_original_first(tmp_path):
    """Shim wraps the user's adaptive_fn; user's function runs to completion
    before ROME does its corpus sweep. Crucial because IMPRESS's adaptive_fn
    updates pipeline.current_scores — ROME's sweep relies on the CSV, not
    the in-memory dict, but the ordering needs to be predictable.
    """
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec, train_batch_threshold=999)

    call_order: List[str] = []

    async def user_adaptive(pipeline):
        call_order.append("user")
        return None

    wrapped = shim.wrap_adaptive_fn(user_adaptive)
    pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
    _write_csv(
        str(tmp_path / "af_stats_p1_pass_0.csv"),
        [{"ID": "b1.pdb", "plddt": 85.0, "ptm": 0.85, "pae": 3.5}],
    )

    asyncio.run(wrapped(pipe))
    assert call_order == ["user"]
    assert shim.corpus_size == 1


@pytest.mark.fast
def test_shim_with_no_user_adaptive_fn(tmp_path):
    """``wrap_adaptive_fn(None)`` returns a callable that only does ROME's
    sweep — handy for users who want corpus + training without IMPRESS's
    sub-pipeline-spawn logic.
    """
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec, train_batch_threshold=999)
    wrapped = shim.wrap_adaptive_fn(None)
    pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
    _write_csv(
        str(tmp_path / "af_stats_p1_pass_0.csv"),
        [{"ID": "b1.pdb", "plddt": 85.0, "ptm": 0.85, "pae": 3.5}],
    )
    asyncio.run(wrapped(pipe))
    assert shim.corpus_size == 1


@pytest.mark.fast
def test_corpus_admits_only_passing_rows(tmp_path):
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec, train_batch_threshold=999)
    wrapped = shim.wrap_adaptive_fn(None)
    pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
    _write_csv(
        str(tmp_path / "af_stats_p1_pass_0.csv"),
        [
            {"ID": "good.pdb", "plddt": 90.0, "ptm": 0.90, "pae": 3.0},  # passes
            {"ID": "low_plddt.pdb", "plddt": 70.0, "ptm": 0.85, "pae": 3.0},  # fails pLDDT
            {"ID": "low_ptm.pdb", "plddt": 85.0, "ptm": 0.50, "pae": 3.0},    # fails pTM
            {"ID": "high_pae.pdb", "plddt": 85.0, "ptm": 0.85, "pae": 7.0},   # fails pAE
        ],
    )
    asyncio.run(wrapped(pipe))
    assert shim.corpus_size == 1


@pytest.mark.fast
def test_training_fires_when_threshold_crossed(tmp_path):
    """Two passes, each adds 2 qualifying entries -> threshold of 2 trips
    on pass 0; should fire training and bump model_version to 1.
    """
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec, train_batch_threshold=2)

    async def run():
        manager = FakeManager()
        wrapped = shim.wrap_adaptive_fn(None)

        def writer(p):
            _write_csv(
                str(tmp_path / f"af_stats_{p.name}_pass_{p.passes}.csv"),
                [
                    {"ID": f"b1_{p.passes}.pdb", "plddt": 90.0, "ptm": 0.9, "pae": 3.0},
                    {"ID": f"b2_{p.passes}.pdb", "plddt": 88.0, "ptm": 0.88, "pae": 3.5},
                ],
            )

        pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
        async with shim.attached(manager) as r:
            await manager.run_passes(pipe, n_passes=2, csv_writer=writer, adaptive_fn=wrapped)
            return r

    r = asyncio.run(run())
    assert r.training_rounds >= 1
    assert r.model_version >= 1
    assert r.current_checkpoint is not None
    assert "weights.pt" in os.listdir(r.current_checkpoint)
    assert rec.calls, "trainer hook was not called"


@pytest.mark.fast
def test_csv_dedup_across_repeated_reads(tmp_path):
    """If the same CSV is read more than once (e.g. extract re-run), entries
    are added once. seq_uid tracking dedupes."""
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec, train_batch_threshold=999)
    wrapped = shim.wrap_adaptive_fn(None)
    pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
    _write_csv(
        str(tmp_path / "af_stats_p1_pass_0.csv"),
        [{"ID": "b1.pdb", "plddt": 90.0, "ptm": 0.9, "pae": 3.0}],
    )
    asyncio.run(wrapped(pipe))
    asyncio.run(wrapped(pipe))  # second sweep, same CSV
    assert shim.corpus_size == 1


@pytest.mark.fast
def test_missing_csv_is_silent(tmp_path):
    """If no CSV exists yet (e.g. pre-first-pass), the shim is a no-op
    rather than an error. Matches IMPRESS's tolerance for first passes
    that haven't produced scores."""
    rec = TrainRecorder()
    shim = _make_shim(tmp_path, rec)
    wrapped = shim.wrap_adaptive_fn(None)
    pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
    # No CSV written.
    asyncio.run(wrapped(pipe))
    assert shim.corpus_size == 0
    assert rec.calls == []


@pytest.mark.fast
def test_attached_awaits_in_flight_training(tmp_path):
    """``async with shim.attached(manager):`` should not exit until any
    in-flight training tasks finish — otherwise the rest of the user's
    teardown could outrun the trainer and break observability."""
    rec = TrainRecorder()

    train_started = asyncio.Event() if False else None  # marker

    async def slow_train(config, sampled, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "weights.pt"), "wb") as fd:
            fd.write(b"\x00")
        await asyncio.sleep(0.05)  # simulate training latency
        rec.calls.append({"shard_size": len(sampled), "ckpt": ckpt_dir})
        return ckpt_dir

    shim = RomeShim(
        corpus_thresholds=CorpusThresholds(),
        train_batch_threshold=1,
        train_shard_size=4,
        mpnn_checkpoint_dir=str(tmp_path / "ckpts"),
        base_path=str(tmp_path),
        task_hooks=TaskHooks(mpnn_train=slow_train),
        csv_path_for=lambda p: str(tmp_path / f"af_stats_{p.name}_pass_{p.passes}.csv"),
    )

    async def run():
        wrapped = shim.wrap_adaptive_fn(None)
        pipe = FakeProteinBindingPipeline(name="p1", af_out_path=str(tmp_path))
        _write_csv(
            str(tmp_path / "af_stats_p1_pass_0.csv"),
            [{"ID": "b1.pdb", "plddt": 90.0, "ptm": 0.9, "pae": 3.0}],
        )
        async with shim.attached(None):
            await wrapped(pipe)
            # training is in flight here; the context manager exit must wait

    asyncio.run(run())
    # If attached() returned before training finished, rec.calls would be empty.
    assert len(rec.calls) == 1
    assert shim.model_version == 1
