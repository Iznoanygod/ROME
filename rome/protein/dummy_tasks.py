"""In-memory dummy science-tool hooks for orchestration tests.

These are *not* used in production — they're swap-ins so the flow can be
exercised end-to-end without foundry / AF2 / IMPRESS scripts installed. Each
dummy:

* matches the signature its real counterpart in :mod:`rome.protein.tasks`
  exposes,
* records what it was called with (for assertions),
* synthesizes plausible outputs so the orchestration's branches all fire.

Use :func:`make_dummy_hooks` to assemble a ready-to-use ``TaskHooks``.
"""

import asyncio
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from rome.protein.hooks import TaskHooks
from rome.protein.schema import AF2Result, SequenceRecord


# ---------------------------------------------------------------------------
# Configurable score generator — lets tests force a specific cycle pattern
# ---------------------------------------------------------------------------

@dataclass
class ScorePlan:
    """How AF2 scores should evolve per backbone, per cycle.

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
    af2_calls: List[Dict[str, Any]] = field(default_factory=list)
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
                # generate samples_per_loop sequences with decreasing log-likelihood
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


def make_dummy_af2(recorder: DummyRecorder):
    """No-op AF2 that just records the call and creates the output dir."""

    async def af2(config, fasta_dir, fasta_filename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        recorder.af2_calls.append(
            {
                "fasta_dir": fasta_dir,
                "fasta_filename": fasta_filename,
                "output_dir": output_dir,
            }
        )
        return output_dir

    return af2


def make_dummy_extract(recorder: DummyRecorder, score_plan: ScorePlan):
    """Synthesize one AF2Result per call using the score plan."""

    async def extract(config, pipeline_id, cycle, af_output_dir, csv_out_path):
        # backbone_id is encoded by the flow into the FASTA path — but the
        # flow also tags it on the result post-extract, so the dummy just
        # has to emit *something*. We attach a placeholder backbone_id;
        # the flow rewrites it before recording.
        seq_uid = os.path.basename(af_output_dir)
        # In production the extractor inspects multiple model PDBs; here
        # one row per call is sufficient.
        # Score plan lookup uses the encoded backbone_id from seq_uid
        # ("<backbone_id>-w<n>-c<n>-<hash>")
        backbone_id = seq_uid.split("-")[0] if "-" in seq_uid else seq_uid
        plddt, ptm, pae = score_plan.get(backbone_id, cycle)
        result = AF2Result(
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
        # produce a sentinel "weight file" so callers can verify the path
        with open(os.path.join(output_checkpoint_dir, "weights.pt"), "wb") as fd:
            fd.write(b"\x00")
        return output_checkpoint_dir

    return train


def make_dummy_hooks(
    recorder: DummyRecorder,
    score_plan: ScorePlan,
    samples_per_loop: int = 4,
) -> TaskHooks:
    """Bundle the four dummy hooks into a TaskHooks ready for the flow."""
    return TaskHooks(
        mpnn_generator_loop=make_dummy_mpnn_generator(recorder, samples_per_loop),
        af2_predict=make_dummy_af2(recorder),
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
