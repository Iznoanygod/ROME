"""Entrypoint mirroring IMPRESS's ``run_protein_binding.py``.

Streaming MPNN + adaptive AF2 + continuous MPNN training. Paths to the
science tools (foundry MPNN weights, AF2 script + container + DBs, IMPRESS
extract script) come in via ``ProteinBindingFlowConfig``; ROME does not
modify or vendor those tools.

Drop input PDBs under ``./structures/`` (or override ``structures=`` below)
and run on a node with GPUs that match the resource block at the top.
"""

import asyncio
import os
from pathlib import Path

from radical.asyncflow import RadicalExecutionBackend  # type: ignore

from rome.flows.proteinbindingflow import ProteinBindingFlow
from rome.protein import BackboneSpec, ProteinBindingFlowConfig


HERE = Path(__file__).parent
STRUCTURES_DIR = HERE / "structures"


def _discover_backbones() -> list[BackboneSpec]:
    """Pick up every .pdb under ./structures/ as a backbone."""
    specs = []
    for p in sorted(STRUCTURES_DIR.glob("*.pdb")):
        specs.append(
            BackboneSpec(
                backbone_id=p.stem,
                pdb_path=str(p),
                # Match the paper's PDZ + Alpha-Synuclein-tail setup: design
                # chain A (receptor), leave the peptide chain alone.
                design_chains="A",
            )
        )
    return specs


async def main() -> None:
    backend = await RadicalExecutionBackend(
        {
            "gpus": int(os.environ.get("ROME_GPUS", 4)),
            "cores": int(os.environ.get("ROME_CORES", 32)),
            "runtime": int(os.environ.get("ROME_RUNTIME_MIN", 23 * 60)),
            "resource": os.environ.get("ROME_RESOURCE", "purdue.anvil_gpu"),
        }
    )

    config = ProteinBindingFlowConfig(
        structures=_discover_backbones(),
        # L1 — streaming generators
        num_mpnn_generators=2,
        mpnn_batch_size=4,
        seqs_per_mpnn_call=10,
        max_buffer_per_backbone=64,
        ll_top_k_per_backbone=4,
        # L2 — adaptive cycles
        num_af2_workers=2,
        max_cycles=4,
        max_fallback_sequences=10,
        max_sub_pipelines=3,
        # Training (continuous)
        train_mpnn=True,
        min_pLDDT_for_corpus=80.0,
        min_pTM_for_corpus=0.8,
        max_pAE_for_corpus=5.0,
        train_batch_threshold=64,
        train_shard_size=256,
        train_sampling="uniform",
        # Backends — point these at your local installs
        mpnn_backend="foundry",
        mpnn_weights_dir=os.environ.get("MPNN_WEIGHTS_DIR"),
        mpnn_checkpoint_dir=os.environ.get("MPNN_CKPT_DIR"),
        af2_script=os.environ.get("AF2_SCRIPT"),
        af2_image=os.environ.get("AF2_IMAGE"),
        af2_db_root=os.environ.get("AF2_DB_ROOT"),
        extract_script=os.environ.get("EXTRACT_SCRIPT"),
        base_path=os.environ.get("ROME_BASE_PATH", "./protein_binding_run"),
    )

    flow = ProteinBindingFlow(config=config, asyncflow=backend)
    await flow.launch()
    await backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
