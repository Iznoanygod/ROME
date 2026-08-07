import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from oldrome.protein.schema import BackboneSpec


@dataclass
class ProteinBindingPipeline:
    pipeline_id: str
    base_path: str
    backbones: Dict[str, BackboneSpec]
    is_child: bool = False
    start_cycle: int = 0
    passes: int = 0
    sub_order: int = 0
    seq_rank: int = 0
    # Per-backbone fallback counter (L2 escalation budget consumed so far).
    fallback_attempts: Dict[str, int] = field(default_factory=dict)
    # Per-backbone current / previous AF2 score dicts.
    current_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    previous_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Backbones still being worked on (key matches BackboneSpec.backbone_id).
    iter_seqs: Dict[str, BackboneSpec] = field(default_factory=dict)
    kill_parent: bool = False

    # ------------------------------------------------------------------
    # sandbox layout
    # ------------------------------------------------------------------
    @property
    def input_path(self) -> str:
        return os.path.join(self.base_path, f"{self.pipeline_id}_in")

    @property
    def mpnn_out_path(self) -> str:
        return os.path.join(self.base_path, f"{self.pipeline_id}_mpnn")

    @property
    def af_out_path(self) -> str:
        return os.path.join(self.base_path, f"{self.pipeline_id}_af")

    def set_up_dirs(self) -> None:
        for p in (self.input_path, self.mpnn_out_path, self.af_out_path):
            os.makedirs(p, exist_ok=True)

    def stats_csv(self, cycle: Optional[int] = None) -> str:
        """Per-pass score CSV.

        Matches IMPRESS's adaptive_decision which reads
        ``af_stats_<pipeline_name>_pass_<pass>.csv``.
        """
        c = self.passes if cycle is None else cycle
        return os.path.join(
            self.af_out_path, f"af_stats_{self.pipeline_id}_pass_{c}.csv"
        )

    # ------------------------------------------------------------------
    # adaptive sub-pipeline helpers
    # ------------------------------------------------------------------

    def migrate_backbones(self, backbone_ids: List[str]) -> Dict[str, BackboneSpec]:
        """Pop a subset of backbones from this pipeline's working set.

        Returns the popped specs so the caller can hand them to a fresh
        child pipeline.
        """
        moved = {}
        for bid in backbone_ids:
            spec = self.iter_seqs.pop(bid, None)
            if spec is not None:
                moved[bid] = spec
        return moved

    def copy_pdbs_into(self, child_input_dir: str, backbone_ids: List[str]) -> None:
        """Copy the latest AF-predicted PDBs into a child's input dir.

        IMPRESS uses the best AF2 model as the next backbone. We mirror
        that: ``<af_out_path>/<backbone_id>.pdb`` is the convention the
        extractor writes for the highest-ranked prediction this cycle.
        """
        os.makedirs(child_input_dir, exist_ok=True)
        for bid in backbone_ids:
            src = os.path.join(self.af_out_path, f"{bid}.pdb")
            if not os.path.exists(src):
                continue
            dst = os.path.join(child_input_dir, f"{bid}.pdb")
            shutil.copyfile(src, dst)

    def to_state(self) -> Dict[str, Any]:
        """Serialize to a plain dict for the workflow ddict."""
        return {
            "pipeline_id": self.pipeline_id,
            "base_path": self.base_path,
            "is_child": self.is_child,
            "start_cycle": self.start_cycle,
            "passes": self.passes,
            "sub_order": self.sub_order,
            "seq_rank": self.seq_rank,
            "fallback_attempts": dict(self.fallback_attempts),
            "current_scores": dict(self.current_scores),
            "previous_scores": dict(self.previous_scores),
            "iter_seqs": dict(self.iter_seqs),
            "kill_parent": self.kill_parent,
        }