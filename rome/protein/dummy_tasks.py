"""In-memory dummy science-tool hooks for orchestration tests.

These are *not* used in production — they're swap-ins so the flow can be
exercised end-to-end without foundry / AF2 / IMPRESS scripts installed.
Each dummy:

* matches the signature its real counterpart in :mod:`rome.protein.tasks`
  exposes,
* records what it was called with (for assertions),
* synthesizes plausible outputs so the orchestration's branches all fire.

Use :func:`make_dummy_hooks` to assemble a ready-to-use ``TaskHooks``.

Contracts match the IMPRESS main-branch AF2 pipeline:
  * ``make_dummy_mpnn_generator``   → ProteinMPNN streaming
  * ``make_dummy_predict``          → AF2-multimer (3-arg signature)
  * (stage_prediction)              → defaults to no-op; not stubbed here
  * ``make_dummy_extract``          → plddt_extract_pipeline.py
  * ``make_dummy_train``            → ROME training round
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
                # back-pressure: throttle on the ranked queue length since
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
    """AF2 stand-in: creates the output dir + a sentinel PDB so the
    extractor has something to scan. Matches af2_multimer_reduced.sh's
    three-arg signature.
    """

    async def predict(config, fasta_dir, fasta_filename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Seq UID is the FASTA stem (the flow writes ``<seq_uid>.fasta``).
        seq_uid = fasta_filename.rsplit(".", 1)[0]
        # Sentinel PDB at <output_dir>/<seq_uid>.pdb so _parse_extract_csv
        # can resolve a plausible pdb_path. The dummy extractor synthesizes
        # the scores; the file contents don't need to be valid.
        with open(os.path.join(output_dir, f"{seq_uid}.pdb"), "w") as fd:
            fd.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
            )
        recorder.predict_calls.append(
            {
                "fasta_dir": fasta_dir,
                "fasta_filename": fasta_filename,
                "output_dir": output_dir,
            }
        )
        return output_dir

    return predict


def make_dummy_extract(recorder: DummyRecorder, score_plan: ScorePlan):
    """Synthesize one PredictionResult per call using the score plan.

    The dummy reads the seq_uid out of the AF output dir (which the
    predict dummy named after the FASTA stem) and decodes the backbone
    id from its ``<backbone_id>-w<n>-c<n>-<hash>`` shape — same encoding
    ``make_dummy_mpnn_generator`` produces.
    """

    async def extract(config, pipeline_id, cycle, af_output_dir, csv_out_path):
        seq_uid = os.path.basename(af_output_dir)
        backbone_id = seq_uid.split("-")[0] if "-" in seq_uid else seq_uid
        plddt, ptm, pae = score_plan.get(backbone_id, cycle)
        result = PredictionResult(
            seq_uid=seq_uid,
            backbone_id=backbone_id,
            pdb_path=os.path.join(af_output_dir, f"{seq_uid}.pdb"),
            pLDDT=plddt,
            pTM=ptm,
            pAE=pae,
        )
        recorder.extract_calls.append(
            {
                "pipeline_id": pipeline_id,
                "cycle": cycle,
                "result": result.__dict__,
            }
        )
        return [result]

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
    """Bundle the dummy hooks into a TaskHooks ready for the flow.

    ``stage_prediction`` is left unset — the production default
    (``rome.protein.tasks.stage_prediction_task``) is a no-op pass-through,
    which is what AF2 needs.
    """
    return TaskHooks(
        mpnn_generator_loop=make_dummy_mpnn_generator(recorder, samples_per_loop),
        predict_structure=make_dummy_predict(recorder),
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
