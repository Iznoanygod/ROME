"""Science-tool integration seams.

Each task is a thin shell over an *external* tool — foundry's ProteinMPNN
engine, AlphaFold2 (containerized), the IMPRESS pLDDT/pTM/pAE extractor,
and foundry's MPNN trainer. Per the project constraint, none of the
science code is modified or vendored here; we shell out or lazy-import.

These are pure async functions so they can be wrapped with
``asyncflow.function_task`` / ``executable_task`` decorators at orchestrator
construction time without baking the engine reference into module import.
"""

import asyncio
import csv
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from rome.protein.schema import AF2Result, BackboneSpec, SequenceRecord


# ---------------------------------------------------------------------------
# MPNN generation (streaming worker body)
# ---------------------------------------------------------------------------

async def mpnn_generate_loop(
    config: Any,
    worker_index: int,
    workflow_ddict: Any,
    terminate_event: Any,
) -> None:
    """Streaming generator: continuously samples ProteinMPNN until termination.

    The worker:
    * loads the MPNN inference engine pointed at ``config.mpnn_weights_dir``
    * in a loop, picks a backbone whose ranked buffer has slack
      (< ``max_buffer_per_backbone`` candidates) and runs one batch
    * appends ``SequenceRecord``-shaped dicts to
      ``workflow_ddict["mpnn_outputs"][backbone_id]``
    * between batches, checks ``model_version`` and reloads weights if newer

    Foundry / legacy adapter selection is the only branch — both end up
    producing ``(sequence, log_likelihood)`` pairs.
    """
    engine, current_version = _load_mpnn_engine(config, workflow_ddict)

    while not terminate_event.is_set():
        backbone_id = _pick_backbone(config, workflow_ddict)
        if backbone_id is None:
            await asyncio.sleep(0.1)
            continue

        spec: BackboneSpec = workflow_ddict["backbone_pool"][backbone_id]
        current_version, engine = _maybe_reload_mpnn(
            engine, config, workflow_ddict, current_version
        )

        records = _mpnn_sample_batch(
            engine=engine,
            spec=spec,
            num_seqs=config.seqs_per_mpnn_call,
            model_version=current_version,
            cycle=workflow_ddict.get("global_cycle", 0),
        )
        _append_mpnn_outputs(workflow_ddict, backbone_id, records)


def _load_mpnn_engine(config: Any, workflow_ddict: Any):
    """Lazy-load foundry's MPNNInferenceEngine (or a legacy shim).

    Imports happen inside the function so ``rome.protein`` stays importable
    in environments without foundry installed. The engine handle is opaque
    to the rest of the orchestration code.
    """
    version = workflow_ddict.get("model_version", 0)
    path = workflow_ddict.get("mpnn_checkpoint_path") or config.mpnn_weights_dir

    if config.mpnn_backend == "foundry":
        # Foundry's MPNN package is API-stabilizing; import lazily.
        from mpnn.inference_engines import MPNNInferenceEngine  # type: ignore

        engine = MPNNInferenceEngine.from_pretrained(path)
        return engine, version

    if config.mpnn_backend == "legacy":
        return _LegacyMPNNEngine(path), version

    raise ValueError(f"Unknown mpnn_backend: {config.mpnn_backend}")


def _maybe_reload_mpnn(engine, config, workflow_ddict, local_version):
    """Hot-swap weights when the trainer has bumped the version."""
    remote = workflow_ddict.get("model_version", 0)
    if remote <= local_version:
        return local_version, engine
    new_path = workflow_ddict.get("mpnn_checkpoint_path")
    if new_path is None:
        return local_version, engine
    if config.mpnn_backend == "foundry":
        from mpnn.inference_engines import MPNNInferenceEngine  # type: ignore

        engine = MPNNInferenceEngine.from_pretrained(new_path)
    elif config.mpnn_backend == "legacy":
        engine = _LegacyMPNNEngine(new_path)
    return remote, engine


def _pick_backbone(config: Any, workflow_ddict: Any) -> Optional[str]:
    """Round-robin / slack-based selection over the active backbone pool."""
    pool = workflow_ddict.get("backbone_pool", {})
    ranked = workflow_ddict.get("ranked_candidates", {}) or {}
    for bid in pool:
        bucket = ranked.get(bid, [])
        if len(bucket) < config.max_buffer_per_backbone:
            return bid
    return None


def _mpnn_sample_batch(
    engine,
    spec: BackboneSpec,
    num_seqs: int,
    model_version: int,
    cycle: int,
) -> List[Dict[str, Any]]:
    """Run one ProteinMPNN sample call against ``spec`` and return records.

    The contract the engine must satisfy:
      engine.sample(pdb_path, num_seqs, design_chains, fixed_positions, ...)
        -> Iterable[(sequence: str, log_likelihood: float)]

    Both foundry's MPNNInferenceEngine and the legacy shim conform to it.
    """
    raw = engine.sample(
        pdb_path=spec.pdb_path,
        num_seqs=num_seqs,
        design_chains=spec.design_chains,
        fixed_positions=spec.fixed_positions,
        bias_AA=spec.bias_AA,
        bias_weight=spec.bias_weight,
        temperature=spec.temp,
    )
    records = []
    for seq, ll in raw:
        records.append(
            SequenceRecord(
                seq_uid=str(uuid.uuid4()),
                backbone_id=spec.backbone_id,
                sequence=seq,
                log_likelihood=float(ll),
                produced_under_version=model_version,
                produced_in_cycle=cycle,
            ).__dict__
        )
    return records


def _append_mpnn_outputs(workflow_ddict: Any, backbone_id: str, records: list) -> None:
    outputs = workflow_ddict.get("mpnn_outputs", {}) or {}
    bucket = outputs.get(backbone_id, [])
    bucket.extend(records)
    outputs[backbone_id] = bucket
    workflow_ddict["mpnn_outputs"] = outputs


class _LegacyMPNNEngine:
    """Adapter for dauparas/ProteinMPNN-style CLI (what IMPRESS uses).

    Not a re-implementation — shells out to the original repo's scripts.
    """

    def __init__(self, weights_path: str):
        self.weights_path = weights_path

    def sample(self, *args, **kwargs):  # pragma: no cover - integration seam
        raise NotImplementedError(
            "Legacy MPNN sampling shells out to dauparas/ProteinMPNN's "
            "protein_mpnn_run.py — wire your local mpnn_wrapper.py here."
        )


# ---------------------------------------------------------------------------
# AlphaFold2 prediction
# ---------------------------------------------------------------------------

async def af2_predict_task(
    config: Any,
    fasta_dir: str,
    fasta_filename: str,
    output_dir: str,
) -> str:
    """Run AlphaFold2 multimer on a single FASTA. Returns the output dir.

    Wraps a script following IMPRESS's ``af2_multimer_reduced.sh`` interface:
        <script> <fasta_dir> <fasta_filename> <output_dir>
    The script + container + database paths are taken from ``config``.
    """
    if not config.af2_script:
        raise ValueError("config.af2_script must be set for af2_predict_task")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [config.af2_script, fasta_dir, fasta_filename, output_dir]
    env = os.environ.copy()
    if config.af2_image:
        env["AF2_IMAGE"] = config.af2_image
    if config.af2_db_root:
        env["AF2_DB_ROOT"] = config.af2_db_root
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"af2_script failed (rc={proc.returncode}): {stderr.decode()[:2000]}"
        )
    return output_dir


# ---------------------------------------------------------------------------
# Extract pLDDT / pTM / pAE
# ---------------------------------------------------------------------------

async def extract_metrics_task(
    config: Any,
    pipeline_id: str,
    cycle: int,
    af_output_dir: str,
    csv_out_path: str,
) -> List[AF2Result]:
    """Run IMPRESS's pLDDT/pTM/pAE extractor and parse its CSV.

    Shells out to ``config.extract_script`` with arguments matching
    ``plddt_extract_pipeline.py``:
        <script> --iteration <cycle> --name <pipeline_id> --base <af_output_dir>
            --out <csv_out_path>
    """
    if not config.extract_script:
        raise ValueError("config.extract_script must be set for extract_metrics_task")
    cmd = [
        config.extract_script,
        "--iteration", str(cycle),
        "--name", pipeline_id,
        "--base", af_output_dir,
        "--out", csv_out_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"extract_script failed (rc={proc.returncode}): {stderr.decode()[:2000]}"
        )
    return _parse_extract_csv(csv_out_path)


def _parse_extract_csv(csv_path: str) -> List[AF2Result]:
    """Parse the CSV emitted by IMPRESS's extractor: ID, avg_plddt, ptm, avg_pae."""
    results: List[AF2Result] = []
    with open(csv_path) as fd:
        reader = csv.reader(fd)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            ident, plddt, ptm, pae = row[0], row[1], row[2], row[3]
            # seq_uid is encoded by callers into the ID; backbone_id parsed from prefix
            seq_uid = ident
            backbone_id = ident.split(".")[0]
            results.append(
                AF2Result(
                    seq_uid=seq_uid,
                    backbone_id=backbone_id,
                    pdb_path=os.path.join(os.path.dirname(csv_path), f"{ident}.pdb"),
                    pLDDT=float(plddt),
                    pTM=float(ptm),
                    pAE=float(pae),
                    raw_csv_row=",".join(row),
                )
            )
    return results


# ---------------------------------------------------------------------------
# MPNN training
# ---------------------------------------------------------------------------

async def mpnn_train_task(
    config: Any,
    sampled_entries: list,
    output_checkpoint_dir: str,
) -> str:
    """Invoke foundry's MPNN trainer on a curated shard of corpus entries.

    ``sampled_entries`` is a list of ``CorpusEntry``-shaped dicts already
    drawn from the full corpus by the flow's sampler. This implementation
    materializes them into a parquet shard (foundry's expected input) and
    invokes the trainer.

    Returns the path to the newly written checkpoint so the orchestrator
    can publish it via ``workflow_ddict["mpnn_checkpoint_path"]`` and bump
    ``model_version``.
    """
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    shard_path = _write_parquet_shard(config, sampled_entries)

    # Lazy import — foundry trainer pulls in lightning fabric + atomworks.
    from mpnn.trainers.mpnn import MPNNTrainer  # type: ignore

    trainer = MPNNTrainer(
        train_data=shard_path,
        output_dir=output_checkpoint_dir,
        **config.mpnn_train_config,
    )
    # Trainer is sync; run it in a thread so we don't block the event loop.
    await asyncio.to_thread(trainer.fit)
    return output_checkpoint_dir


def _write_parquet_shard(config: Any, entries: list) -> str:
    """Materialize ``entries`` as a parquet file under config.base_path."""
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    shard_dir = os.path.join(config.base_path, "training_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"shard_{uuid.uuid4().hex[:8]}.parquet")
    table = pa.Table.from_pylist(entries)
    pq.write_table(table, shard_path)
    return shard_path
