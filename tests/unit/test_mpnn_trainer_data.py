"""ProteinMPNN trainer: everything reachable without foundry, atomworks or a GPU.

The trainer has two halves. The *data* half — validate, build_training_dataframe,
write_shard — turns a ROME-A corpus into the parquet metadata frame foundry
consumes, and needs nothing but pandas. The *training* half (`_train_with_foundry`)
imports foundry/atomworks/torch and runs on a GPU; it is exercised nowhere here
and cannot be, since none of those import in this environment.

So these tests answer three of the questions a campaign operator actually has:

  * "will it crash on my data?" — feed campaign-shaped records, including the
    awkward ones, through the data half and require clear failures, not deep ones.
  * "what data do I need?" — the required-field and column tests pin it down.
  * "is it built for the right ProteinMPNN?" — see the docstrings on
    test_shard_is_foundry_format and docs/proteinmpnn_training.md. The trainer
    targets foundry's re-implementation; IMPRESS runs the original dauparas CLI.
    That mismatch is a decision, not something a test can paper over, but the
    format tests at least make the target explicit.
"""

import os

import pandas as pd
import pytest

from rome.train.mpnn import (
    DEFAULT_FILTERS,
    PEPTIDE_MAX_RESIDUES,
    ProteinMPNNConfig,
    ProteinMPNNTrainer,
    build_training_dataframe,
)


# --- fixtures: a structure file on disk, shaped like an IMPRESS AF output ----

_MONOMER_PDB = """\
ATOM      1  N   MET A   1      0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  MET A   1      1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   MET A   1      2.009   1.420   0.000  1.00 90.00           C
END
"""

# A designed PDZ domain (chain A) plus the fixed 10-mer peptide (chain B) — the
# actual shape of an IMPRESS protein-binding AF prediction.
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


def _design(tmp_path, uid, *, backbone="p1", seq="MKTAYIAKQR", pdb=_DIMER_PDB,
            plddt=95.0, ptm=0.90, pae=3.5, **extra):
    """A corpus record shaped like one an IMPRESS-R adaptive_fn contributes."""
    record = {
        "uid": uid,
        "backbone_id": backbone,
        "path": _write(tmp_path, f"{uid}.pdb", pdb),
        "sequence": seq,
        "pLDDT": plddt,
        "pTM": ptm,
        "pAE": pae,
    }
    record.update(extra)
    return record


# --- what data the trainer needs --------------------------------------------

def test_required_fields_are_path_and_sequence(tmp_path):
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    trainer.validate([_design(tmp_path, "d1")])          # complete: fine


def test_a_sequence_without_a_structure_is_rejected():
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    with pytest.raises(ValueError, match="path"):
        trainer.validate([{"sequence": "MKV"}])


def test_validate_scans_every_record_not_just_the_first(tmp_path):
    """The failure that motivated fixing validate: a bad record deep in the list."""
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    corpus = [_design(tmp_path, f"d{i}") for i in range(6)]
    del corpus[4]["path"]                                # the fifth is broken
    with pytest.raises(ValueError, match=r"record 4 is missing.*path"):
        trainer.validate(corpus)


# --- the dataframe foundry consumes -----------------------------------------

def test_dataframe_has_every_column_the_parser_and_weighting_read(tmp_path):
    frame = build_training_dataframe([_design(tmp_path, "d1")])
    # GenericDFParser
    for column in ("example_id", "path", "assembly_id"):
        assert column in frame.columns, column
    # filters + token-budget batching
    assert "n_non_atomized_tokens" in frame.columns
    # AF3-style weighting (n_peptide is read unguarded — a miss is a bare KeyError)
    for column in ("cluster", "n_prot", "n_peptide", "n_nuc", "n_ligand",
                   "q_pn_unit_is_loi"):
        assert column in frame.columns, column


def test_path_is_absolute_so_the_worker_finds_it(tmp_path):
    """The training worker runs in another process/node; a relative path breaks."""
    frame = build_training_dataframe([_design(tmp_path, "d1")])
    assert os.path.isabs(frame.iloc[0]["path"])


def test_token_count_falls_back_to_sequence_length(tmp_path):
    frame = build_training_dataframe([_design(tmp_path, "d1", seq="A" * 84)])
    assert frame.iloc[0]["n_non_atomized_tokens"] == 84


def test_clustering_keeps_one_backbone_from_dominating(tmp_path):
    frame = build_training_dataframe([
        _design(tmp_path, "d1", backbone="p1"),
        _design(tmp_path, "d2", backbone="p1"),
        _design(tmp_path, "d3", backbone="p2"),
    ])
    assert list(frame["cluster"]) == ["p1", "p1", "p2"]


def test_provenance_survives_into_the_shard(tmp_path):
    """The shard is the audit trail for what a round trained on."""
    frame = build_training_dataframe([_design(tmp_path, "d1", model_version=4)])
    row = frame.iloc[0]
    assert row["pLDDT"] == 95.0 and row["pTM"] == 0.90 and row["pAE"] == 3.5
    assert row["produced_under_version"] == 4


# --- write_shard: what a real round does before it ever touches foundry ------

def test_write_shard_round_trips_through_parquet(tmp_path):
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    corpus = [_design(tmp_path, "d1"), _design(tmp_path, "d2")]
    shard = trainer.write_shard(corpus, str(tmp_path / "round"))
    back = pd.read_parquet(shard)
    assert list(back["design_id"]) == ["d1", "d2"]
    # Every column must survive the parquet round-trip with a usable dtype;
    # an all-None object column, for instance, would trip pyarrow.
    assert back["n_non_atomized_tokens"].dtype.kind in "iu"


def test_a_whole_campaign_group_builds_without_crashing(tmp_path):
    """176 records with the score spread of a real PDZ group — the smoke test."""
    corpus = []
    for i in range(176):
        corpus.append(_design(
            tmp_path, f"d{i}",
            backbone=f"p{i % 16}",
            seq="A" * (76 + i % 33),                     # 76..108, the real range
            plddt=88.0 + (i % 10), ptm=0.67 + (i % 29) / 100,
            pae=2.0 + (i % 6),
        ))
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    trainer.validate(corpus)
    shard = trainer.write_shard(corpus, str(tmp_path / "round"))
    assert len(pd.read_parquet(shard)) == 176


# --- the two things a campaign operator most needs to see -------------------

def test_dimer_designs_expose_the_n_prot_contradiction(tmp_path):
    """IMPRESS designs are dimers, but the dataframe hardcodes n_prot == 1.

    This is not a passing-is-good test; it documents a known inconsistency so a
    change to it is deliberate. The structure file has two chains (designed PDZ +
    peptide), yet build_training_dataframe writes n_prot == 1, and the default
    filter *requires* n_prot == 1. So today the dimer sails through the filter
    while its metadata disagrees with its coordinates — see
    docs/proteinmpnn_training.md. Whichever way this is resolved, this test
    should change with it.
    """
    frame = build_training_dataframe([_design(tmp_path, "d1", pdb=_DIMER_PDB)])
    assert frame.iloc[0]["n_prot"] == 1                  # metadata says monomer...
    # ...and the default filter admits only monomers, so a *correct* n_prot == 2
    # would be silently filtered out. Both facts are load-bearing.
    assert "n_prot == 1" in DEFAULT_FILTERS


def test_shard_is_foundry_format_not_original_proteinmpnn(tmp_path):
    """Pin the implementation target so a mismatch is visible, not latent.

    The shard is a *dataframe of structure-file paths*, which is foundry's
    training input. The original dauparas ProteinMPNN trains from per-chain .pt
    tensors and a clustering CSV, and loads/saves checkpoints as
    {'model_state_dict': ...}. IMPRESS runs that original at inference. A
    checkpoint produced from this shard therefore will NOT load into IMPRESS's
    MPNN without a foundry->legacy conversion that does not exist yet.
    """
    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    shard = trainer.write_shard([_design(tmp_path, "d1")], str(tmp_path / "r"))
    columns = set(pd.read_parquet(shard).columns)
    # foundry's GenericDFParser signature...
    assert {"example_id", "path", "assembly_id"} <= columns
    # ...and specifically not the original repo's per-chain tensor layout.
    assert "seq_chain_A" not in columns and "coords_chain_A" not in columns


def test_short_designs_are_flagged_as_peptides(tmp_path):
    frame = build_training_dataframe([
        _design(tmp_path, "short", seq="A" * PEPTIDE_MAX_RESIDUES),
        _design(tmp_path, "long", seq="A" * (PEPTIDE_MAX_RESIDUES + 1)),
    ])
    assert list(frame["n_peptide"]) == [1, 0]
