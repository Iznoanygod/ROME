"""Data records exchanged between protein-binding pipeline stages.

These are plain dataclasses — no domain libraries imported here so they can
travel through Dragon DDict without dragging heavy deps onto worker nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackboneSpec:
    """A single input structure plus its design constraints.

    Constraints mirror the flags accepted by IMPRESS's ``mpnn_wrapper.py`` so
    a backbone can be handed to ProteinMPNN without further translation. The
    ``target_peptide`` field carries the binding-partner sequence appended to
    the paired FASTA the structure predictor (Boltz / AF2) consumes — in the
    IMPRESS PDZ use case this is the C-terminal Alpha-Synuclein peptide.
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
    # Paired-FASTA fields (consumed in s3). The designed sequence is written
    # under ``>designed_chain_name|<backbone_id>``; the target peptide under
    # ``>target_chain_name|<backbone_id>``. Boltz/AF2 fold the resulting
    # dimer. Defaults match the IMPRESS branch layout (pdz + pep).
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

    Container for the per-cycle structure-prediction metrics — pLDDT from
    backbone B-factors, iPTM+PTM from the predictor's confidence JSON, and
    cross-interface PAE. Works for either Boltz or AF2 outputs since the
    fields are the predictor-agnostic scoring surface IMPRESS's
    plddt_extract_pipeline.py produces.
    """
    seq_uid: str
    backbone_id: str
    pdb_path: str
    pLDDT: float
    pTM: float
    pAE: float
    raw_csv_row: Optional[str] = None


# Back-compat alias for the AF2-era name. Drop once external callers settle.
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
