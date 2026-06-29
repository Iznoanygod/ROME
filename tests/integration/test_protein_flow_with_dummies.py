"""End-to-end orchestration test for :class:`ProteinBindingFlow`.

Uses the in-memory state factory and the dummy task hooks from
:mod:`rome.protein.dummy_tasks`. No Dragon, no RADICAL backend, no foundry,
no AF2, no IMPRESS extractor — but every orchestration branch (streaming
generation, L1 ranking, AF2 + extract per cycle, criterion decisions, corpus
gating, training trigger, weight-version bump) is exercised.

Drop-in production wiring: replace ``make_dummy_hooks(...)`` with
``TaskHooks()`` (or a partially-populated one) and ``state_factory`` with
the Dragon default.

Contracts match the IMPRESS main-branch AF2-multimer pipeline.
"""

import asyncio
import os

import pytest

from rome.flows.proteinbindingflow import ProteinBindingFlow
from rome.protein import (
    BackboneSpec,
    ProteinBindingFlowConfig,
)
from rome.protein.dummy_tasks import (
    DummyRecorder,
    ScorePlan,
    in_memory_state_factory,
    make_dummy_hooks,
)


def _backbones(tmp_path, names):
    specs = []
    for n in names:
        p = tmp_path / f"{n}.pdb"
        p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
        specs.append(BackboneSpec(backbone_id=n, pdb_path=str(p)))
    return specs


def _cfg(tmp_path, structures, **over):
    base = dict(
        structures=structures,
        num_mpnn_generators=1,
        mpnn_batch_size=2,
        seqs_per_mpnn_call=4,
        max_buffer_per_backbone=8,
        ll_top_k_per_backbone=2,
        num_af2_workers=1,
        max_cycles=2,
        max_fallback_sequences=2,
        max_sub_pipelines=2,
        train_mpnn=True,
        min_pLDDT_for_corpus=80.0,
        min_pTM_for_corpus=0.8,
        max_pAE_for_corpus=5.0,
        train_batch_threshold=2,
        train_shard_size=4,
        base_path=str(tmp_path / "run"),
    )
    base.update(over)
    return ProteinBindingFlowConfig(**base)


@pytest.mark.fast
def test_flow_runs_end_to_end_on_dummies(tmp_path):
    """All hooks called, cycles complete, FASTAs written."""
    structures = _backbones(tmp_path, ["b1", "b2"])
    recorder = DummyRecorder()
    plan = ScorePlan(default=(85.0, 0.85, 3.5))  # always-improving → all KEEPs

    flow = ProteinBindingFlow(
        config=_cfg(tmp_path, structures, max_cycles=2),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    # MPNN generators produced something for every backbone
    sampled_bids = {s["backbone_id"] for s in recorder.mpnn_samples}
    assert sampled_bids == {"b1", "b2"}

    # Predict + extract each ran at least max_cycles per backbone
    assert len(recorder.predict_calls) >= 2 * 2
    assert len(recorder.extract_calls) >= 2 * 2

    # Predict was called with the AF2 three-arg signature
    for call in recorder.predict_calls:
        assert "fasta_dir" in call
        assert "fasta_filename" in call
        assert "output_dir" in call
        assert call["fasta_filename"].endswith(".fasta")

    # Cycle results recorded in the ddict
    cycle_results = flow._workflow_ddict["cycle_results"]
    for bid in ("b1", "b2"):
        assert len(cycle_results[bid]) == 2  # two cycles


@pytest.mark.fast
def test_paired_fasta_emitted_when_target_peptide_set(tmp_path):
    """When BackboneSpec.target_peptide is set the flow writes both the
    designed sequence record and the peptide record (Boltz-compatible).
    AF2-multimer pipelines that don't set target_peptide get a single
    sequence record.
    """
    structures = _backbones(tmp_path, ["b1"])
    structures[0].target_peptide = "EGYQDYEPEA"
    recorder = DummyRecorder()
    plan = ScorePlan(default=(85.0, 0.85, 3.5))

    flow = ProteinBindingFlow(
        config=_cfg(tmp_path, structures, max_cycles=1, train_mpnn=False),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    import glob
    fastas = glob.glob(str(tmp_path / "run" / "*_mpnn" / "*.fasta"))
    assert fastas
    text = open(fastas[0]).read()
    assert ">pdz|" in text and ">pep|" in text
    assert "EGYQDYEPEA" in text


@pytest.mark.fast
def test_corpus_grows_and_training_fires(tmp_path):
    """High-confidence cycles populate corpus, training hook fires + bumps version."""
    structures = _backbones(tmp_path, ["b1"])
    recorder = DummyRecorder()
    plan = ScorePlan(default=(90.0, 0.9, 3.0))  # passes thresholds

    flow = ProteinBindingFlow(
        config=_cfg(
            tmp_path,
            structures,
            max_cycles=3,
            train_batch_threshold=1,  # fire on the very first eligible cycle
        ),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    # At least one training round happened
    assert len(recorder.train_calls) >= 1
    # Each training shard saw at least one entry
    assert all(c["shard_size"] >= 1 for c in recorder.train_calls)
    # The model_version in the ddict bumped at least as many times
    assert flow._workflow_ddict["model_version"] >= 1
    # Checkpoint path now points at the trainer's output
    assert flow._workflow_ddict["mpnn_checkpoint_path"] is not None
    assert "weights.pt" in os.listdir(
        flow._workflow_ddict["mpnn_checkpoint_path"]
    )


@pytest.mark.fast
def test_corpus_excludes_low_quality_entries(tmp_path):
    """Cycles failing thresholds should NOT seed the training corpus."""
    structures = _backbones(tmp_path, ["b1"])
    recorder = DummyRecorder()
    # pTM below floor → never qualifies
    plan = ScorePlan(default=(85.0, 0.5, 3.0))

    flow = ProteinBindingFlow(
        config=_cfg(
            tmp_path, structures,
            max_cycles=2,
            train_batch_threshold=1,
        ),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    assert recorder.train_calls == []
    assert flow._workflow_ddict["model_version"] == 0
    assert flow._workflow_ddict["corpus"] == {}


@pytest.mark.fast
def test_degraded_backbone_spawns_sub_pipeline(tmp_path):
    """A regressing backbone exhausts fallbacks then migrates to a child."""
    structures = _backbones(tmp_path, ["b1"])
    recorder = DummyRecorder()
    # cycle 0 good, cycle 1 regresses → fallback budget burns then MIGRATE
    plan = ScorePlan(
        per_backbone={
            "b1": [
                (90.0, 0.9, 3.0),   # baseline
                (70.0, 0.6, 6.0),   # degraded
            ]
        },
        default=(70.0, 0.6, 6.0),
    )

    flow = ProteinBindingFlow(
        config=_cfg(
            tmp_path, structures,
            max_cycles=2,
            max_fallback_sequences=1,
            max_sub_pipelines=1,
            train_mpnn=False,
        ),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    # Sub-pipeline was created → predict ran more than just (1 backbone * 2 cycles)
    # since fallback re-runs predict within cycle 1, and the migrated child runs too.
    assert len(recorder.predict_calls) >= 3


@pytest.mark.fast
def test_per_backbone_score_isolation(tmp_path):
    """Extract returns rows for every backbone in the staging dir; the
    flow must attribute each row to the correct backbone — not
    overwrite all rows with the currently-resolving backbone's id.

    Regression for a bug where the flow rewrote ``backbone_id`` on
    every row returned by extract, causing a regressing backbone to
    inherit an improving one's scores.
    """
    structures = _backbones(tmp_path, ["good", "bad"])
    recorder = DummyRecorder()
    # "good" improves; "bad" stays low. They share a staging dir, so
    # extract returns rows for both each time it's called.
    plan = ScorePlan(
        per_backbone={
            "good": [(85.0, 0.85, 3.5), (88.0, 0.88, 3.0)],
            "bad":  [(85.0, 0.85, 3.5), (60.0, 0.50, 7.0)],  # cycle 1 regresses
        },
    )

    flow = ProteinBindingFlow(
        config=_cfg(tmp_path, structures, max_cycles=2, train_mpnn=False),
        task_hooks=make_dummy_hooks(recorder, plan),
        state_factory=in_memory_state_factory,
    )
    asyncio.run(flow.launch())

    cycle_results = flow._workflow_ddict["cycle_results"]

    def score_at(bid, cycle):
        for s in cycle_results[bid]:
            if s["cycle"] == cycle and s["backbone_id"] == bid:
                return s["prediction"]["pLDDT"]
        return None

    # The improving backbone's recorded scores match its score plan, not
    # the regressing one's.
    assert score_at("good", 0) == 85.0
    assert score_at("good", 1) == 88.0
    # The regressing backbone keeps its own (low) scores; it does NOT
    # inherit "good"'s 88.0.
    assert score_at("bad", 0) == 85.0
    bad_cycle1 = score_at("bad", 1)
    # Either it stayed at 60 (no fallback budget yet) or migrated (and
    # therefore was recorded under a child pipeline with the bad score),
    # but in no case should it be 88.0.
    assert bad_cycle1 != 88.0


@pytest.mark.fast
def test_default_stage_prediction_is_passthrough(tmp_path):
    """AF2 doesn't need post-predict staging. The production default for
    ``stage_prediction`` is a no-op that returns the predict output dir
    unchanged — non-AF2 predictors (e.g. Boltz) override the hook to
    inject chain renames or file restructuring.
    """
    import asyncio as _asyncio
    from rome.protein.tasks import stage_prediction_task

    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    (predict_dir / "ranked_0.pdb").write_text("ATOM\n")

    out = _asyncio.run(
        stage_prediction_task(
            config=None,
            prediction_output_dir=str(predict_dir),
            target_fasta="anything.fasta",
            backbone_id="b1",
        )
    )
    # No-op: returned the predict dir unchanged, didn't touch contents.
    assert out == str(predict_dir)
    assert (predict_dir / "ranked_0.pdb").exists()


@pytest.mark.fast
def test_swap_in_real_tool_only_overrides_unset_hooks(tmp_path):
    """A partially-populated TaskHooks falls through to defaults for missing slots."""
    from rome.protein.hooks import TaskHooks
    from rome.protein import tasks as real_tasks

    recorder = DummyRecorder()
    plan = ScorePlan(default=(85.0, 0.85, 3.5))
    hooks = make_dummy_hooks(recorder, plan)
    # Wipe one hook so the default kicks in
    partial = TaskHooks(
        mpnn_generator_loop=hooks.mpnn_generator_loop,
        predict_structure=hooks.predict_structure,
        extract_metrics=hooks.extract_metrics,
        mpnn_train=None,
    )
    resolved = partial.resolved()
    assert resolved.mpnn_generator_loop is hooks.mpnn_generator_loop
    assert resolved.predict_structure is hooks.predict_structure
    assert resolved.mpnn_train is real_tasks.mpnn_train_task
    # stage_prediction falls back to the production no-op pass-through.
    assert resolved.stage_prediction is real_tasks.stage_prediction_task
