"""Configuration for :class:`rome.flows.proteinbindingflow.ProteinBindingFlow`.

Single dataclass split into logical blocks (design / L1 streaming / L2 adaptive /
training / backends / resources). No ML library imports.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from rome.protein.schema import BackboneSpec


@dataclass
class ProteinBindingFlowConfig:
    # ---- backbones ----
    structures: List[BackboneSpec] = field(default_factory=list)

    # ---- L1: streaming MPNN generation + log-likelihood admission ----
    num_mpnn_generators: int = 2
    mpnn_batch_size: int = 4
    seqs_per_mpnn_call: int = 10
    max_buffer_per_backbone: int = 64
    ll_top_k_per_backbone: int = 4

    # ---- L2: adaptive promotion / sub-pipeline spawning ----
    num_af2_workers: int = 2
    max_cycles: int = 4
    max_fallback_sequences: int = 10
    max_sub_pipelines: int = 3
    # Default adaptive criterion lives in rome.protein.criteria; this is the
    # plug point. Signature: async (curr: dict[str,float], prev: dict[str,float],
    # backbone_id: str, cfg) -> Decision (see criteria.py).
    adaptive_criterion: Optional[Callable] = None

    # ---- corpus / training feedback loop ----
    train_mpnn: bool = True
    min_pLDDT_for_corpus: float = 80.0
    min_pTM_for_corpus: float = 0.8
    max_pAE_for_corpus: float = 5.0
    # Trigger a training round when this many new entries have accumulated since
    # the last fired training task.
    train_batch_threshold: int = 64
    train_max_concurrent: int = 1
    train_shard_size: int = 256
    train_sampling: str = "uniform"          # 'uniform' | 'weighted_by_score'
    mpnn_train_config: Dict[str, Any] = field(default_factory=dict)

    # ---- backends ----
    # 'foundry' uses MPNNInferenceEngine; 'legacy' shells out to dauparas/ProteinMPNN.
    mpnn_backend: str = "foundry"
    mpnn_weights_dir: Optional[str] = None       # initial checkpoint
    mpnn_checkpoint_dir: Optional[str] = None    # where trainer writes new versions

    # AF2 — shell out to a script following af2_multimer_reduced.sh's signature:
    #   <script> <fasta_dir> <fasta_filename> <output_dir>
    af2_script: Optional[str] = None
    af2_image: Optional[str] = None
    af2_db_root: Optional[str] = None

    # Extract — shell out to IMPRESS's plddt_extract_pipeline.py-equivalent.
    extract_script: Optional[str] = None

    # ---- resources (RADICAL backend) ----
    gpus_per_mpnn_task: int = 1
    gpus_per_af2_task: int = 1
    gpus_per_train_task: int = 1
    cpus_per_extract_task: int = 1

    # ---- sandbox layout ----
    base_path: str = "./protein_binding_run"
