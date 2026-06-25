"""Runnable demo of :class:`ProteinBindingFlow` against print+sleep stand-ins.

Drives the **full** orchestration end-to-end — streaming MPNN generation,
log-likelihood ranking, structure prediction, staging, extraction, the
adaptive criterion (KEEP / FALLBACK / MIGRATE / DROP), corpus
accumulation, training, weight-version hot reload — without any real
science tools.  Every hook prints what it's doing and ``await
asyncio.sleep(...)`` to simulate the duration of the real work it stands
in for.

Run it with::

    python examples/protein_binding/demo_run.py

Output looks like::

    [+0.00s] [BOOT]          backbones=['b_alpha', 'b_beta', 'b_degraded'] cycles=3
    [+0.00s] [s1 MPNN]       w0 batch for b_alpha cycle=0 v0 (5 seqs)
    [+0.00s] [s1 MPNN]       w1 batch for b_alpha cycle=0 v0 (5 seqs)
    [+0.31s] [s4 Boltz]      predict b_alpha (fasta b_alpha.fa)
    [+0.82s] [s4_post]       stage b_alpha -> best_models / best_ptm + rename chains
    [+0.88s] [s5 extract]    pid=p_root_... cycle=0
    ...

The scenario is configured so:
  * two backbones ("b_alpha", "b_beta") improve every cycle (KEEPs),
  * a third ("b_degraded") regresses on cycle 1, exhausting fallback
    budget and migrating to a sub-pipeline (MIGRATE branch),
  * the corpus crosses ``train_batch_threshold`` mid-run, firing the
    trainer hook, which bumps ``model_version``; the streaming MPNN
    workers print the hot-reload event between batches.

To wire production tools, swap the hook bodies for what's in
``rome.protein.tasks`` and run the same flow under a Dragon-backed
state factory.
"""

import asyncio
import os
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

# Make the demo runnable without ``pip install -e .`` — add the repo root
# (two levels up from this file) to sys.path before importing rome.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ``rome/__init__.py`` eagerly imports the LLM trainer stack (trl + peft +
# transformers + datasets). The protein flow doesn't need any of it, so we
# stub these out before importing — same trick ``tests/conftest.py`` plays
# so the protein orchestration is exercisable on a minimal Python env.
def _stub_llm_modules():
    """Stub the same modules ``tests/conftest.py`` does so the protein
    flow is importable on a minimal Python env (no torch/peft/dragon/...).
    """
    import types

    def ensure(name: str, attrs: dict | None = None) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        try:
            return __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            for k, v in (attrs or {}).items():
                setattr(mod, k, v)
            sys.modules[name] = mod
            return mod

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    ensure("torch", {"no_grad": _NoGrad})
    ensure("trl", {"GRPOConfig": object, "SFTConfig": object, "GRPOTrainer": object})
    ensure(
        "peft",
        {
            "get_peft_model": lambda m, c: m,
            "LoraConfig": object,
            "PeftModel": object,
        },
    )
    ensure(
        "transformers",
        {
            "GenerationConfig": object,
            "AutoTokenizer": object,
            "AutoModelForCausalLM": object,
        },
    )
    ensure("datasets", {"Dataset": object})
    ensure("radical", {})
    ensure("radical.asyncflow", {"WorkflowEngine": object})
    ensure("dragon", {})
    ensure("dragon.data", {})
    ddict_mod = ensure("dragon.data.ddict", {"DDict": object})
    sys.modules["dragon.data"].ddict = ddict_mod
    ensure("dragon.native", {})
    event_mod = ensure("dragon.native.event", {"Event": object})
    sys.modules["dragon.native"].event = event_mod
    ensure("rose", {})
    ensure(
        "rose.metrics",
        {"GREATER_THAN_THRESHOLD": "greater_than_threshold"},
    )
    ensure(
        "rose.learner",
        {
            "SequentialReinforcementLearner": type(
                "SequentialReinforcementLearner",
                (),
                {"__init__": lambda self, asyncflow=None: None},
            )
        },
    )


_stub_llm_modules()

from rome.flows.proteinbindingflow import ProteinBindingFlow
from rome.protein import (
    BackboneSpec,
    PredictionResult,
    ProteinBindingFlowConfig,
    SequenceRecord,
    TaskHooks,
)
from rome.protein.dummy_tasks import in_memory_state_factory


# ---------------------------------------------------------------------------
# logging helpers — wall-clock-relative timestamps make the streaming +
# concurrency visible in the output.
# ---------------------------------------------------------------------------

DEMO_START = time.monotonic()


def _t() -> str:
    return f"[+{time.monotonic() - DEMO_START:5.2f}s]"


def _log(stage: str, msg: str) -> None:
    print(f"{_t()} {stage:<14} {msg}", flush=True)


# ---------------------------------------------------------------------------
# scenario knobs — separated out so they're easy to twiddle when reading
# ---------------------------------------------------------------------------

# Backbone that should regress on cycle 1 so the FALLBACK -> MIGRATE
# escalation path runs.
REGRESS_BID = "b_degraded"

# Per-stage simulated work durations. Tuned so a full run completes in
# ~15-20 seconds while still feeling like things are happening.
SLEEP_MPNN = 0.30      # one ProteinMPNN batch on GPU
SLEEP_PREDICT = 0.50   # one Boltz prediction
SLEEP_STAGE = 0.05     # cp + chain rename
SLEEP_EXTRACT = 0.10   # PyRosetta parse
SLEEP_TRAIN = 1.50     # one fine-tuning round


# ---------------------------------------------------------------------------
# s1 — streaming ProteinMPNN
# ---------------------------------------------------------------------------

async def demo_mpnn_loop(config, worker_index, ddict, terminate):
    """Continuously samples MPNN until termination. Mirrors the production
    body in ``rome.protein.tasks.mpnn_generate_loop`` but with prints +
    sleeps instead of a real engine call.
    """
    version = ddict.get("model_version", 0)
    rng = random.Random(0xDEC0DE + worker_index)
    while not terminate.is_set():
        pool = ddict.get("backbone_pool", {}) or {}
        ranked = ddict.get("ranked_candidates", {}) or {}
        outputs = ddict.get("mpnn_outputs", {}) or {}
        produced_anywhere = False

        for bid in pool:
            if len(ranked.get(bid, [])) >= config.max_buffer_per_backbone:
                continue
            cycle = ddict.get("global_cycle", 0)
            _log(
                "[s1 MPNN]",
                f"w{worker_index} batch for {bid} cycle={cycle} v{version} "
                f"({config.seqs_per_mpnn_call} seqs)",
            )
            await asyncio.sleep(SLEEP_MPNN)  # simulate GPU sampling
            bucket = outputs.get(bid, [])
            for i in range(config.seqs_per_mpnn_call):
                rec = SequenceRecord(
                    seq_uid=f"{bid}-w{worker_index}-c{cycle}-{uuid.uuid4().hex[:6]}",
                    backbone_id=bid,
                    sequence="M" + "".join(
                        rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(19)
                    ),
                    log_likelihood=-1.0 - i * 0.1,
                    produced_under_version=version,
                    produced_in_cycle=cycle,
                ).__dict__
                bucket.append(rec)
            outputs[bid] = bucket
            produced_anywhere = True

        ddict["mpnn_outputs"] = outputs

        # hot-reload check between batches — the streamflow weight-sync
        # pattern, reused by the protein flow
        new_v = ddict.get("model_version", version)
        if new_v != version:
            _log(
                "[s1 MPNN]",
                f"w{worker_index} hot-reload v{version} -> v{new_v} "
                f"(ckpt={ddict.get('mpnn_checkpoint_path')})",
            )
            version = new_v

        if not produced_anywhere:
            await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# s4 — structure prediction
# ---------------------------------------------------------------------------

async def demo_predict(config, fasta_path, output_dir):
    bid = Path(fasta_path).stem
    _log("[s4 Boltz]", f"predict {bid} (fasta {Path(fasta_path).name})")
    await asyncio.sleep(SLEEP_PREDICT)
    nested = Path(output_dir) / f"boltz_results_{bid}" / "predictions" / bid
    nested.mkdir(parents=True, exist_ok=True)
    # Write a Boltz-shaped output tree so the next stage (s4_post_exec) has
    # something realistic to stage + chain-rename.
    (nested / f"{bid}_model_0.pdb").write_text(
        "ATOM      1  CA  ALA pdz   1       0.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      2  CA  ALA pep   1       1.000   1.000   1.000  1.00 90.00           C\n"
    )
    (nested / f"confidence_{bid}_model_0.json").write_text(
        '{"ptm": 0.85, "iptm": 0.80}\n'
    )
    _log("[s4 Boltz]", f"predict {bid} done -> {nested.name}/")
    return output_dir


# ---------------------------------------------------------------------------
# s4_post_exec — stage outputs into canonical paths + rename chains
# ---------------------------------------------------------------------------

async def demo_stage(config, predict_out, best_model_dst, best_ptm_dst, bid):
    _log("[s4_post]", f"stage {bid} -> best_models/ + best_ptm/ (chain rename pdz->A pep->B)")
    await asyncio.sleep(SLEEP_STAGE)
    src = Path(predict_out) / f"boltz_results_{bid}" / "predictions" / bid
    Path(best_model_dst).parent.mkdir(parents=True, exist_ok=True)
    Path(best_ptm_dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / f"{bid}_model_0.pdb", best_model_dst)
    shutil.copyfile(src / f"confidence_{bid}_model_0.json", best_ptm_dst)
    return best_model_dst


# ---------------------------------------------------------------------------
# s5 — pLDDT/pTM/pAE extraction
# ---------------------------------------------------------------------------

async def demo_extract(config, pipeline_id, cycle, prediction_root, csv_out):
    """Synthesize improving scores by default; degrade REGRESS_BID on cycle 1+."""
    _log("[s5 extract]", f"pid={pipeline_id} cycle={cycle}")
    await asyncio.sleep(SLEEP_EXTRACT)
    best_models = Path(prediction_root) / "best_models"
    results = []
    for fn in sorted(best_models.glob("*.pdb")):
        bid = fn.stem
        if bid == REGRESS_BID and cycle >= 1:
            plddt, ptm, pae = 72.0, 0.65, 5.5    # degraded → triggers fallback
        else:
            plddt = min(92.0, 80.0 + cycle * 3.0)
            ptm = min(0.95, 0.80 + cycle * 0.04)
            pae = max(2.5, 4.5 - cycle * 0.5)
        results.append(
            PredictionResult(
                seq_uid=bid,
                backbone_id=bid,
                pdb_path=str(fn),
                pLDDT=plddt,
                pTM=ptm,
                pAE=pae,
            )
        )
        _log(
            "[s5 extract]",
            f"  {bid:<12} pLDDT={plddt:5.1f}  pTM={ptm:.2f}  pAE={pae:.2f}",
        )
    return results


# ---------------------------------------------------------------------------
# MPNN training (ROME extension)
# ---------------------------------------------------------------------------

async def demo_train(config, sampled, ckpt_dir):
    _log("[TRAIN]", f"start fine-tune shard={len(sampled)} entries -> {ckpt_dir}")
    await asyncio.sleep(SLEEP_TRAIN)
    os.makedirs(ckpt_dir, exist_ok=True)
    # sentinel "weights" file so anyone inspecting the run can see it
    (Path(ckpt_dir) / "weights.pt").write_bytes(b"\x00")
    _log("[TRAIN]", f"done; new checkpoint = {ckpt_dir}")
    return ckpt_dir


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

async def main():
    base = Path("/tmp/rome_demo")
    if base.exists():
        shutil.rmtree(base)
    structures_dir = base / "structures"
    structures_dir.mkdir(parents=True)

    bids = ["b_alpha", "b_beta", REGRESS_BID]
    specs = []
    for bid in bids:
        p = structures_dir / f"{bid}.pdb"
        p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
        specs.append(
            BackboneSpec(
                backbone_id=bid,
                pdb_path=str(p),
                design_chains="A",
                target_peptide="EGYQDYEPEA",
            )
        )

    cfg = ProteinBindingFlowConfig(
        structures=specs,
        # streaming generators
        num_mpnn_generators=2,
        seqs_per_mpnn_call=5,
        max_buffer_per_backbone=8,
        # adaptive cycles — tight budget so MIGRATE fires quickly
        max_cycles=3,
        max_fallback_sequences=1,
        max_sub_pipelines=1,
        # corpus + training — low threshold so we see at least one
        # fine-tune round during the demo
        train_mpnn=True,
        min_pLDDT_for_corpus=80.0,
        min_pTM_for_corpus=0.80,
        max_pAE_for_corpus=5.0,
        train_batch_threshold=2,
        train_shard_size=4,
        base_path=str(base / "run"),
    )

    hooks = TaskHooks(
        mpnn_generator_loop=demo_mpnn_loop,
        predict_structure=demo_predict,
        stage_prediction=demo_stage,
        extract_metrics=demo_extract,
        mpnn_train=demo_train,
    )

    _log("[BOOT]", f"backbones={bids} cycles={cfg.max_cycles}")
    _log("[BOOT]", f"sandbox={cfg.base_path}")
    flow = ProteinBindingFlow(
        config=cfg,
        task_hooks=hooks,
        state_factory=in_memory_state_factory,
    )
    await flow.launch()
    _log("[DONE]", "all pipelines drained")

    _print_summary(flow)


def _print_summary(flow) -> None:
    d = flow._workflow_ddict
    corpus = d.get("corpus", {})
    cycle_res = d.get("cycle_results", {})

    print()
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"  total wall time:   {time.monotonic() - DEMO_START:5.2f}s")
    print(f"  final model_version: {d.get('model_version', 0)}")
    print(f"  mpnn_checkpoint_path: {d.get('mpnn_checkpoint_path')}")
    print(f"  corpus size:       {len(corpus)} pairs")
    print(f"  cycle results per backbone:")
    for bid, summaries in cycle_res.items():
        scored = [
            (s["cycle"], s["prediction"]["pLDDT"], s["prediction"]["pTM"], s["prediction"]["pAE"])
            for s in summaries
        ]
        print(f"    {bid:<12} {len(summaries)} cycles | "
              + " | ".join(f"c{c}: pLDDT={p:5.1f} pTM={t:.2f} pAE={e:.2f}"
                           for c, p, t, e in scored))
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
