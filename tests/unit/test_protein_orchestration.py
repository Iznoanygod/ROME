"""Unit tests for the protein-binding orchestration scaffolding.

These exercise the criterion + coordinator + corpus logic without touching
any science tools (no foundry MPNN, no AF2, no extract script). The
science seams in ``rome.protein.tasks`` are intentionally not covered here
— they're integration points against external installs.
"""
import asyncio

import pytest

from rome.protein.config import ProteinBindingFlowConfig
from rome.protein.coordinator import AdaptiveCoordinator
from rome.protein.criteria import (
    CriterionInput,
    Decision,
    default_criterion,
)
from rome.protein.pipeline import ProteinBindingPipeline
from rome.protein.ranker import LogLikelihoodRanker
from rome.protein.schema import BackboneSpec


def _cfg(**over):
    base = dict(
        max_cycles=4,
        max_fallback_sequences=2,
        max_sub_pipelines=2,
    )
    base.update(over)
    return ProteinBindingFlowConfig(**base)


def test_default_criterion_first_pass_keeps():
    inp = CriterionInput(
        backbone_id="b1",
        current={"pLDDT": 80, "pTM": 0.8, "pAE": 4.0},
        previous=None,
        fallback_attempts=0,
        sub_order=0,
        config=_cfg(),
    )
    assert asyncio.run(default_criterion(inp)) == Decision.KEEP


def test_default_criterion_keeps_when_improved():
    inp = CriterionInput(
        backbone_id="b1",
        current={"pLDDT": 85, "pTM": 0.9, "pAE": 3.5},
        previous={"pLDDT": 80, "pTM": 0.85, "pAE": 4.0},
        fallback_attempts=0,
        sub_order=0,
        config=_cfg(),
    )
    assert asyncio.run(default_criterion(inp)) == Decision.KEEP


def test_default_criterion_falls_back_when_degraded():
    inp = CriterionInput(
        backbone_id="b1",
        current={"pLDDT": 70, "pTM": 0.7, "pAE": 5.0},
        previous={"pLDDT": 80, "pTM": 0.85, "pAE": 4.0},
        fallback_attempts=0,
        sub_order=0,
        config=_cfg(),
    )
    assert asyncio.run(default_criterion(inp)) == Decision.FALLBACK


def test_default_criterion_migrates_when_fallback_exhausted():
    inp = CriterionInput(
        backbone_id="b1",
        current={"pLDDT": 70, "pTM": 0.7, "pAE": 5.0},
        previous={"pLDDT": 80, "pTM": 0.85, "pAE": 4.0},
        fallback_attempts=2,            # equals max_fallback_sequences
        sub_order=0,
        config=_cfg(),
    )
    assert asyncio.run(default_criterion(inp)) == Decision.MIGRATE


def test_default_criterion_drops_when_sub_pipelines_exhausted():
    inp = CriterionInput(
        backbone_id="b1",
        current={"pLDDT": 70, "pTM": 0.7, "pAE": 5.0},
        previous={"pLDDT": 80, "pTM": 0.85, "pAE": 4.0},
        fallback_attempts=2,
        sub_order=2,                    # equals max_sub_pipelines
        config=_cfg(),
    )
    assert asyncio.run(default_criterion(inp)) == Decision.DROP


def test_coordinator_submits_and_completes():
    async def go():
        coord = AdaptiveCoordinator()
        pipe = ProteinBindingPipeline(
            pipeline_id="p_root",
            base_path="/tmp/rome-test",
            backbones={},
        )
        await coord.submit(pipe)
        assert coord.active_count == 1
        nxt = await coord.next_to_run()
        assert nxt is pipe
        await coord.mark_complete(pipe.pipeline_id)
        assert coord.active_count == 0

    asyncio.run(go())


def test_coordinator_spawns_child_with_inherited_state(tmp_path):
    parent = ProteinBindingPipeline(
        pipeline_id="p_root",
        base_path=str(tmp_path),
        backbones={
            "b1": BackboneSpec(backbone_id="b1", pdb_path=str(tmp_path / "b1.pdb")),
            "b2": BackboneSpec(backbone_id="b2", pdb_path=str(tmp_path / "b2.pdb")),
        },
        iter_seqs={
            "b1": BackboneSpec(backbone_id="b1", pdb_path=str(tmp_path / "b1.pdb")),
            "b2": BackboneSpec(backbone_id="b2", pdb_path=str(tmp_path / "b2.pdb")),
        },
        passes=3,
        sub_order=0,
        current_scores={
            "b1": {"pLDDT": 70, "pTM": 0.7, "pAE": 5.0},
            "b2": {"pLDDT": 85, "pTM": 0.9, "pAE": 3.0},
        },
    )
    parent.set_up_dirs()

    moved = parent.migrate_backbones(["b1"])
    coord = AdaptiveCoordinator()
    child = coord.submit_child_pipeline_request(parent, moved)

    assert child.is_child is True
    assert child.sub_order == 1
    assert child.start_cycle == 3
    assert set(child.iter_seqs) == {"b1"}
    assert child.previous_scores["b1"] == parent.current_scores["b1"]
    # parent's iter_seqs no longer contains the migrated backbone
    assert "b1" not in parent.iter_seqs
    assert "b2" in parent.iter_seqs


def test_ranker_sorts_by_log_likelihood_descending():
    async def go():
        ddict = {}
        ddict["mpnn_outputs"] = {
            "b1": [
                {"seq_uid": "s1", "log_likelihood": -2.0},
                {"seq_uid": "s2", "log_likelihood": -0.5},
                {"seq_uid": "s3", "log_likelihood": -1.0},
            ],
        }
        ddict["ranked_candidates"] = {"b1": []}

        class _Ev:
            def __init__(self): self._n = 0
            def is_set(self):
                # let the ranker do exactly one pass, then stop
                self._n += 1
                return self._n > 1
            def set(self): pass

        ranker = LogLikelihoodRanker(poll_interval=0.0)
        await ranker.run(ddict, _Ev())

        ordered = [r["seq_uid"] for r in ddict["ranked_candidates"]["b1"]]
        assert ordered == ["s2", "s3", "s1"]

    asyncio.run(go())


def test_ranker_drains_mpnn_outputs():
    """After a ranker pass, mpnn_outputs is empty so back-pressure works."""
    async def go():
        ddict = {
            "mpnn_outputs": {"b1": [{"seq_uid": "s1", "log_likelihood": -1.0}]},
            "ranked_candidates": {"b1": []},
        }

        class _Ev:
            def __init__(self): self._n = 0
            def is_set(self):
                self._n += 1
                return self._n > 1
            def set(self): pass

        await LogLikelihoodRanker(poll_interval=0.0).run(ddict, _Ev())

        assert ddict["mpnn_outputs"]["b1"] == []
        assert [r["seq_uid"] for r in ddict["ranked_candidates"]["b1"]] == ["s1"]

    asyncio.run(go())


def test_corpus_curator_admits_passing_entries_and_fires_training():
    """Threshold crossing triggers schedule_train_fn exactly once per fill."""
    import asyncio as _asyncio
    from rome.protein.corpus import CorpusCurator

    async def go():
        cfg = _cfg(
            train_mpnn=True,
            min_pLDDT_for_corpus=80.0,
            min_pTM_for_corpus=0.8,
            max_pAE_for_corpus=5.0,
            train_batch_threshold=2,
        )
        ddict = {
            "cycle_results": {
                "b1": [
                    {  # passes
                        "backbone_id": "b1", "cycle": 0, "sequence": "AAA",
                        "produced_under_version": 0,
                        "af2_result": {
                            "seq_uid": "s1", "backbone_id": "b1",
                            "pdb_path": "/tmp/p.pdb",
                            "pLDDT": 85.0, "pTM": 0.85, "pAE": 4.0,
                        },
                    },
                    {  # passes
                        "backbone_id": "b1", "cycle": 1, "sequence": "BBB",
                        "produced_under_version": 0,
                        "af2_result": {
                            "seq_uid": "s2", "backbone_id": "b1",
                            "pdb_path": "/tmp/p.pdb",
                            "pLDDT": 90.0, "pTM": 0.9, "pAE": 3.0,
                        },
                    },
                    {  # fails pTM threshold
                        "backbone_id": "b1", "cycle": 2, "sequence": "CCC",
                        "produced_under_version": 0,
                        "af2_result": {
                            "seq_uid": "s3", "backbone_id": "b1",
                            "pdb_path": "/tmp/p.pdb",
                            "pLDDT": 95.0, "pTM": 0.5, "pAE": 4.0,
                        },
                    },
                ]
            },
            "corpus": {},
            "train_in_flight": False,
        }

        fires = 0
        async def schedule():
            nonlocal fires
            fires += 1

        class _Ev:
            def __init__(self): self._n = 0
            def is_set(self):
                self._n += 1
                return self._n > 1
            def set(self): pass

        curator = CorpusCurator(cfg, schedule, poll_interval=0.0)
        await curator.run(ddict, _Ev())
        # one async task scheduled; await it
        await _asyncio.sleep(0)

        assert len(ddict["corpus"]) == 2
        assert fires == 1
        assert ddict["train_in_flight"] is True

    asyncio.run(go())
