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
    a backbone can be handed to ProteinMPNN without further translation.
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
    # free-form notes the pipeline can attach (e.g. peptide target, source)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceRecord:
    """A single ProteinMPNN-generated sequence ready for AF2 admission."""
    seq_uid: str
    backbone_id: str
    sequence: str
    log_likelihood: float
    produced_under_version: int
    # Cycle that produced the sequence. Distinct from cycle that consumes it.
    produced_in_cycle: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AF2Result:
    """Output of ``af2_predict_task`` + ``extract_metrics_task`` for one sequence."""
    seq_uid: str
    backbone_id: str
    pdb_path: str
    pLDDT: float
    pTM: float
    pAE: float
    raw_csv_row: Optional[str] = None


@dataclass
class CycleResult:
    """Per-(pipeline, backbone, cycle) summary the adaptive coordinator reads."""
    pipeline_id: str
    backbone_id: str
    cycle: int
    seq_uid: str
    af2_result: AF2Result
    # Rank among AF2-evaluated candidates this cycle (0 = top by ll).
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
