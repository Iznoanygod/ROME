"""ProteinMPNN trainer: everything reachable without torch or a GPU.

The trainer has two halves. The *data* half — validate, chain designation,
staging, the manifest, the checkpoint format — turns a ROME-A corpus into what
the original ``dauparas/ProteinMPNN`` fine-tuning loop consumes and publishes,
and needs nothing but the standard library plus pandas. The *training* half
(`_train_with_proteinmpnn`) imports torch and the ProteinMPNN checkout and runs
on a GPU; it is not reachable here and is covered only by being isolated behind
``train_func``.

These answer the questions a campaign operator actually has:

  * "will it crash on my data?" — feed campaign-shaped records, including the
    awkward ones, and require clear early failures, not deep ones.
  * "what data do I need?" — the required-field and designation tests pin it down.
  * "is it the right ProteinMPNN, and does it train dimers?" — the checkpoint is
    the original ``{model_state_dict, num_edges}`` format IMPRESS loads, and the
    chain designation scores the designed chain with the peptide as context.
"""

import os

import pandas as pd
import pytest

from rome.train.mpnn import (
    DEFAULT_CONTEXT_CHAINS,
    DEFAULT_DESIGN_CHAINS,
    ProteinMPNNConfig,
    ProteinMPNNTrainer,
    build_chain_designation,
    original_checkpoint,
    pdb_chain_ids,
    published_weights_path,
    stage_structures,
)


# --- structure files shaped like IMPRESS AF/Boltz outputs --------------------

_MONOMER_PDB = """\
ATOM      1  N   MET A   1      0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  MET A   1      1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   MET A   1      2.009   1.420   0.000  1.00 90.00           C
END
"""

# Designed PDZ domain (chain A) + fixed 10-mer target peptide (chain B) — the
# actual shape of an IMPRESS protein-binding prediction.
_DIMER_PDB = """\
ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  LYS A   2      3.800   0.000   0.000  1.00 90.00           C
ATOM      3  CA  GLU B   1      0.000   3.800   0.000  1.00 88.00           C
END
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _design(tmp_path, uid, *, backbone="p1", seq="MKTAYIAKQR", pdb=_DIMER_PDB, **extra):
    record = {
        "uid": uid,
        "backbone_id": backbone,
        "path": _write(tmp_path, f"{uid}.pdb", pdb),
        "sequence": seq,
        "pLDDT": 95.0, "pTM": 0.90, "pAE": 3.5,
    }
    record.update(extra)
    return record


# --- reading chains out of a structure (no structure library) ----------------

def test_pdb_chain_ids_reads_both_chains_of_a_dimer(tmp_path):
    assert pdb_chain_ids(_write(tmp_path, "d.pdb", _DIMER_PDB)) == ["A", "B"]


def test_pdb_chain_ids_reads_a_monomer(tmp_path):
    assert pdb_chain_ids(_write(tmp_path, "m.pdb", _MONOMER_PDB)) == ["A"]


def test_pdb_chain_ids_tolerates_a_missing_file(tmp_path):
    assert pdb_chain_ids(str(tmp_path / "nope.pdb")) == []


# --- the dimer fix: chain designation ---------------------------------------

def test_dimer_designs_chain_a_with_the_peptide_as_context(tmp_path):
    """The whole point: score the designed chain, keep the peptide as context."""
    designation = build_chain_designation([_design(tmp_path, "d1")])
    designed, context = designation["d1"]
    assert designed == ["A"]                     # binder is designed and scored
    assert context == ["B"]                      # peptide conditions, is not scored
    # And these are the shipped defaults, so a monomer campaign has to opt out.
    assert tuple(DEFAULT_DESIGN_CHAINS) == ("A",)
    assert tuple(DEFAULT_CONTEXT_CHAINS) == ("B",)


def test_designation_drops_chains_absent_from_the_structure(tmp_path):
    """A monomer file under the dimer default: B is simply not there to fix."""
    designation = build_chain_designation([_design(tmp_path, "m", pdb=_MONOMER_PDB)])
    designed, context = designation["m"]
    assert designed == ["A"] and context == []


def test_designation_can_be_overridden_per_record(tmp_path):
    record = _design(tmp_path, "d1", design_chains=["B"], context_chains=["A"])
    designed, context = build_chain_designation([record])["d1"]
    assert designed == ["B"] and context == ["A"]


def test_designation_accepts_a_structure_aware_callable(tmp_path):
    """chains_func decides from the file — e.g. 'design every chain but the last'."""
    def all_but_last(record, present):
        return present[:-1], present[-1:]

    designed, context = build_chain_designation(
        [_design(tmp_path, "d1")], chains_func=all_but_last)["d1"]
    assert designed == ["A"] and context == ["B"]


# --- what data the trainer needs, and failing early --------------------------

def test_validate_accepts_a_well_formed_dimer(tmp_path):
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    trainer.validate([_design(tmp_path, "d1")])


def test_validate_rejects_a_record_without_a_path():
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    with pytest.raises(ValueError, match="path"):
        trainer.validate([{"sequence": "MKV"}])


def test_validate_scans_every_record_not_just_the_first(tmp_path):
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    corpus = [_design(tmp_path, f"d{i}") for i in range(6)]
    del corpus[4]["path"]
    with pytest.raises(ValueError, match=r"record 4 is missing.*path"):
        trainer.validate(corpus)


def test_validate_rejects_a_structure_missing_its_designed_chain(tmp_path):
    """A monomer that only has chain B, under a design-A config, trains nothing."""
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    only_b = "ATOM      1  CA  GLU B   1      0.000   0.000   0.000  1.00 88.00           C\nEND\n"
    with pytest.raises(ValueError, match="designed chains"):
        trainer.validate([_design(tmp_path, "d1", pdb=only_b)])


# --- staging: a stable snapshot, not moving paths ----------------------------

def test_staging_copies_each_structure_under_a_unique_name(tmp_path):
    """Campaign paths collide on basename and get overwritten; staging fixes both."""
    staged = stage_structures(
        [_design(tmp_path, "d1"), _design(tmp_path, "d2")], str(tmp_path / "stage"))
    assert set(staged) == {"d1", "d2"}
    for name, path in staged.items():
        assert os.path.isfile(path) and os.path.basename(path) == f"{name}.pdb"


# --- the manifest a round trains on ------------------------------------------

def test_manifest_carries_scores_and_designation(tmp_path):
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    corpus = [_design(tmp_path, "d1", backbone="p1", model_version=3),
              _design(tmp_path, "d2", backbone="p2")]
    manifest = trainer.write_manifest(corpus, str(tmp_path / "round"))
    frame = pd.read_parquet(manifest)
    assert list(frame["design_id"]) == ["d1", "d2"]
    assert list(frame["designed_chains"]) == ["A", "A"]
    assert list(frame["context_chains"]) == ["B", "B"]
    assert frame.iloc[0]["pLDDT"] == 95.0
    assert frame.iloc[0]["produced_under_version"] == 3


def test_a_whole_campaign_group_builds_without_crashing(tmp_path):
    """176 records at the real PDZ score spread — the smoke test."""
    corpus = [
        _design(tmp_path, f"d{i}", backbone=f"p{i % 16}", seq="A" * (76 + i % 33))
        for i in range(176)
    ]
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    trainer.validate(corpus)
    manifest = trainer.write_manifest(corpus, str(tmp_path / "round"))
    assert len(pd.read_parquet(manifest)) == 176


# --- the implementation target: original ProteinMPNN, not foundry ------------

def test_checkpoint_is_the_format_protein_mpnn_run_loads():
    """protein_mpnn_run.py reads model_state_dict and num_edges; both required."""
    ckpt = original_checkpoint({"layer.weight": 0}, num_edges=48, noise_level=0.2)
    assert set(ckpt) >= {"model_state_dict", "num_edges", "noise_level"}
    # Specifically NOT a foundry/lightning checkpoint ({"model": ...} / "state_dict").
    assert "model" not in ckpt and "state_dict" not in ckpt


def test_weights_land_where_impress_will_load_them(tmp_path):
    """mpnn_wrapper passes no --path_to_model_weights, so the repo default wins."""
    repo = str(tmp_path / "ProteinMPNN")
    cfg = ProteinMPNNConfig(mpnn_repo=repo, model_name="v_48_020",
                            publish_into_repo=True)
    assert published_weights_path(cfg, str(tmp_path / "r")) == \
        os.path.join(repo, "vanilla_model_weights", "v_48_020.pt")
