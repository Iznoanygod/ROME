"""Data records exchanged between protein-binding pipeline stages.

These are plain dataclasses — no domain libraries imported here so they can
travel through Dragon DDict without dragging heavy deps onto worker nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackboneSpec:
    """A single input structure plus its design constraints.

    Constraints mirror the flags accepted by IMPRESS's ``mpnn_wrapper.py``
    so a backbone can be handed to ProteinMPNN without further
    translation.

    The ``target_peptide`` / chain-name fields are optional: they're
    consumed only when running a structure predictor that takes a paired
    FASTA (e.g. Boltz on the ``update_usecase`` branch). The default
    AF2-multimer pipeline on main writes a single-sequence FASTA from the
    designed sequence and ignores these fields.
    """
    backbone_id: str
    pdb_path: str
    is_monomer: bool = False
    design_chains: str = "A"
    fixed_positions: Optional[str] = None
    fix: bool = False
    tied_positions: Optional[str] = None
    homo: bool = False
    bias_AA: Optional[str] = None
    bias_weight: Optional[str] = None
    temp: float = 0.1
    interface: bool = False
    # Optional paired-FASTA fields (only used by non-AF2 predictors).
    target_peptide: Optional[str] = None
    target_chain_name: str = "pep"
    designed_chain_name: str = "pdz"
    # free-form notes the pipeline can attach (e.g. source DB, citation)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceRecord:
    """A single ProteinMPNN-generated sequence ready for prediction admission."""
    seq_uid: str
    backbone_id: str
    sequence: str
    log_likelihood: float
    produced_under_version: int
    # Cycle that produced the sequence. Distinct from cycle that consumes it.
    produced_in_cycle: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Output of ``predict_structure_task`` + ``extract_metrics_task`` for one sequence.

    Container for the per-cycle structure-prediction metrics — mean
    per-residue pLDDT from backbone B-factors, iPTM+PTM from the
    predictor's confidence JSON, and cross-interface PAE. Works for
    AF2-multimer (default) or any drop-in predictor whose extracted
    metrics conform to the IMPRESS ``plddt_extract_pipeline.py`` CSV
    schema.
    """
    seq_uid: str
    backbone_id: str
    pdb_path: str
    pLDDT: float
    pTM: float
    pAE: float
    raw_csv_row: Optional[str] = None


# Back-compat alias matching the AF2-era name used in earlier ROME work.
AF2Result = PredictionResult


@dataclass
class CycleResult:
    """Per-(pipeline, backbone, cycle) summary the adaptive coordinator reads."""
    pipeline_id: str
    backbone_id: str
    cycle: int
    seq_uid: str
    prediction: PredictionResult
    # Rank among predicted candidates this cycle (0 = top by ll).
    fallback_rank: int = 0


@dataclass
class CorpusEntry:
    """(backbone, sequence, metrics) pair eligible for MPNN fine-tuning.

    The corpus is monotonic: entries are never invalidated by later model
    versions. ``produced_under_version`` is metadata only.
    """
    pair_uid: str
    backbone_id: str
    pdb_path: str
    sequence: str
    pLDDT: float
    pTM: float
    pAE: float
    produced_under_version: int
    discovered_at_cycle: int
