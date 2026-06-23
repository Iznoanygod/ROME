"""In-memory dummy science-tool hooks for orchestration tests.

These are *not* used in production — they're swap-ins so the flow can be
exercised end-to-end without foundry / Boltz / AF2 / IMPRESS scripts
installed. Each dummy:

* matches the signature its real counterpart in :mod:`rome.protein.tasks`
  exposes,
* records what it was called with (for assertions),
* synthesizes plausible outputs so the orchestration's branches all fire.

Use :func:`make_dummy_hooks` to assemble a ready-to-use ``TaskHooks``.

Stage mapping (matches IMPRESS update_usecase/protein_binding branch):
  * ``make_dummy_mpnn_generator`` → s1
  * ``make_dummy_predict``        → s4 (Boltz / AF2 stand-in)
  * ``make_dummy_stage``          → s4_post_exec
  * ``make_dummy_extract``        → s5
  * ``make_dummy_train``          → ROME training round
"""

import asyncio
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from rome.protein.hooks import TaskHooks
from rome.protein.schema import PredictionResult, SequenceRecord


# ---------------------------------------------------------------------------
# Configurable score generator — lets tests force a specific cycle pattern
# ---------------------------------------------------------------------------

@dataclass
class ScorePlan:
    """How prediction scores should evolve per backbone, per cycle.

    Each entry is a list of (pLDDT, pTM, pAE) tuples indexed by cycle. If
    the test runs more cycles than entries, the last triple is repeated.
    A scalar ``default`` is used for backbones with no explicit plan.
    """
    per_backbone: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=dict)
    default: Tuple[float, float, float] = (85.0, 0.85, 3.5)

    def get(self, backbone_id: str, cycle: int) -> Tuple[float, float, float]:
        seq = self.per_backbone.get(backbone_id)
        if not seq:
            return self.default
        idx = min(cycle, len(seq) - 1)
        return seq[idx]


# ---------------------------------------------------------------------------
# Recorder — shared spy object the dummy hooks write to
# ---------------------------------------------------------------------------

@dataclass
class DummyRecorder:
    mpnn_samples: List[Dict[str, Any]] = field(default_factory=list)
    predict_calls: List[Dict[str, Any]] = field(default_factory=list)
    stage_calls: List[Dict[str, Any]] = field(default_factory=list)
    extract_calls: List[Dict[str, Any]] = field(default_factory=list)
    train_calls: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def make_dummy_mpnn_generator(recorder: DummyRecorder, samples_per_loop: int = 4):
    """Return a streaming MPNN dummy that produces ranked candidates.

    Log-likelihoods are seeded so the ranker has a deterministic order
    (decreasing index → decreasing ll, so seq_0 ranks highest).
    """

    async def loop(config, worker_index, workflow_ddict, terminate_event):
        rng = random.Random(0xC0FFEE + worker_index)
        version = workflow_ddict.get("model_version", 0)
        while not terminate_event.is_set():
            pool = workflow_ddict.get("backbone_pool", {}) or {}
            outputs = workflow_ddict.get("mpnn_outputs", {}) or {}
            ranked = workflow_ddict.get("ranked_candidates", {}) or {}
            for bid in pool:
                # back-pressure: throttle on the ranked queue length, since
                # the ranker drains mpnn_outputs continuously.
                if len(ranked.get(bid, [])) >= config.max_buffer_per_backbone:
                    continue
                cycle = workflow_ddict.get("global_cycle", 0)
                bucket = outputs.get(bid, [])
                for i in range(samples_per_loop):
                    rec = SequenceRecord(
                        seq_uid=f"{bid}-w{worker_index}-c{cycle}-{uuid.uuid4().hex[:6]}",
                        backbone_id=bid,
                        sequence="A" * 20 + str(rng.randint(0, 9)),
                        log_likelihood=-1.0 - i * 0.1,
                        produced_under_version=version,
                        produced_in_cycle=cycle,
                    ).__dict__
                    bucket.append(rec)
                    recorder.mpnn_samples.append(rec)
                outputs[bid] = bucket
            workflow_ddict["mpnn_outputs"] = outputs
            # bump local version observance — mimics hot reload check
            version = workflow_ddict.get("model_version", version)
            await asyncio.sleep(0.01)

    return loop


def make_dummy_predict(recorder: DummyRecorder):
    """Stand-in for Boltz / AF2. Creates the predictor's nested output
    layout so the staging step has something realistic to copy from.

    Writes:
      <output_dir>/boltz_results_<bid>/predictions/<bid>/<bid>_model_0.pdb
      <output_dir>/boltz_results_<bid>/predictions/<bid>/confidence_<bid>_model_0.json
    """

    async def predict(config, fasta_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Recover the backbone_id from the FASTA filename (<bid>.fa).
        bid = os.path.basename(fasta_path).rsplit(".", 1)[0]
        nested = os.path.join(
            output_dir, f"boltz_results_{bid}", "predictions", bid,
        )
        os.makedirs(nested, exist_ok=True)
        # Synthesize a minimal PDB with multi-char chain IDs so the staging
        # step's chain-rename logic has something to rewrite.
        pdb_path = os.path.join(nested, f"{bid}_model_0.pdb")
        with open(pdb_path, "w") as fd:
            fd.write(
                "ATOM      1  CA  ALA pdz   1       0.000   0.000   0.000  1.00 90.00           C\n"
                "ATOM      2  CA  ALA pep   1       1.000   1.000   1.000  1.00 90.00           C\n"
            )
        json_path = os.path.join(nested, f"confidence_{bid}_model_0.json")
        with open(json_path, "w") as fd:
            fd.write('{"ptm": 0.85, "iptm": 0.80}\n')

        recorder.predict_calls.append(
            {"fasta_path": fasta_path, "output_dir": output_dir, "backbone_id": bid}
        )
        return output_dir

    return predict


def make_dummy_stage(recorder: DummyRecorder):
    """Stand-in for s4_post_exec. Copies the dummy predictor's outputs
    into the canonical staging paths — the same operation the real
    ``stage_prediction_task`` performs (minus the chain rename, which
    on the dummy side reduces to a plain copy since we already know
    the input chain layout).
    """
    import shutil

    async def stage(config, prediction_output_dir, best_model_dst, best_ptm_dst, backbone_id):
        src_root = os.path.join(
            prediction_output_dir,
            f"boltz_results_{backbone_id}",
            "predictions",
            backbone_id,
        )
        src_pdb = os.path.join(src_root, f"{backbone_id}_model_0.pdb")
        src_json = os.path.join(src_root, f"confidence_{backbone_id}_model_0.json")
        os.makedirs(os.path.dirname(best_model_dst), exist_ok=True)
        os.makedirs(os.path.dirname(best_ptm_dst), exist_ok=True)
        if os.path.exists(src_pdb):
            shutil.copyfile(src_pdb, best_model_dst)
        if os.path.exists(src_json):
            shutil.copyfile(src_json, best_ptm_dst)
        recorder.stage_calls.append(
            {
                "prediction_output_dir": prediction_output_dir,
                "best_model_dst": best_model_dst,
                "best_ptm_dst": best_ptm_dst,
                "backbone_id": backbone_id,
            }
        )
        return best_model_dst

    return stage


def make_dummy_extract(recorder: DummyRecorder, score_plan: ScorePlan):
    """Synthesize one PredictionResult per call using the score plan.

    Reads the staging directory to discover which backbones have outputs
    waiting; emits one row per ``best_models/<bid>.pdb`` it finds.
    """

    async def extract(config, pipeline_id, cycle, prediction_root, csv_out_path):
        best_models = os.path.join(prediction_root, "best_models")
        backbone_ids = []
        if os.path.isdir(best_models):
            for fn in sorted(os.listdir(best_models)):
                if fn.endswith(".pdb"):
                    backbone_ids.append(fn[:-4])

        results: List[PredictionResult] = []
        for bid in backbone_ids:
            plddt, ptm, pae = score_plan.get(bid, cycle)
            results.append(
                PredictionResult(
                    seq_uid=bid,  # flow rewrites this with the actual seq_uid
                    backbone_id=bid,
                    pdb_path=os.path.join(best_models, f"{bid}.pdb"),
                    pLDDT=plddt,
                    pTM=ptm,
                    pAE=pae,
                )
            )
        recorder.extract_calls.append(
            {
                "pipeline_id": pipeline_id,
                "cycle": cycle,
                "backbone_ids": list(backbone_ids),
                "csv_out_path": csv_out_path,
            }
        )
        return results

    return extract


def make_dummy_train(recorder: DummyRecorder):
    """No-op trainer — records the shard, returns the checkpoint dir."""

    async def train(config, sampled_entries, output_checkpoint_dir):
        os.makedirs(output_checkpoint_dir, exist_ok=True)
        recorder.train_calls.append(
            {
                "shard_size": len(sampled_entries),
                "output_checkpoint_dir": output_checkpoint_dir,
            }
        )
        with open(os.path.join(output_checkpoint_dir, "weights.pt"), "wb") as fd:
            fd.write(b"\x00")
        return output_checkpoint_dir

    return train


def make_dummy_hooks(
    recorder: DummyRecorder,
    score_plan: ScorePlan,
    samples_per_loop: int = 4,
) -> TaskHooks:
    """Bundle the dummy hooks into a TaskHooks ready for the flow."""
    return TaskHooks(
        mpnn_generator_loop=make_dummy_mpnn_generator(recorder, samples_per_loop),
        predict_structure=make_dummy_predict(recorder),
        stage_prediction=make_dummy_stage(recorder),
        extract_metrics=make_dummy_extract(recorder, score_plan),
        mpnn_train=make_dummy_train(recorder),
    )


# ---------------------------------------------------------------------------
# In-memory state factory (avoids Dragon at test time)
# ---------------------------------------------------------------------------

class InMemoryEvent:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True


def in_memory_state_factory():
    """``(workflow_ddict, terminate_event)`` factory backed by a plain dict."""
    return {}, InMemoryEvent()
