"""Entrypoint mirroring IMPRESS's update_usecase/protein_binding ``run_protein_binding.py``.

Streaming MPNN + adaptive Boltz + continuous MPNN training. Paths to the
science tools (foundry MPNN weights, Boltz/AF2 wrapper scripts, IMPRESS
extract script) come in via ``ProteinBindingFlowConfig``; ROME does not
modify or vendor those tools.

Drop input PDBs under ``./structures/`` (or override ``structures=`` below)
and run on a node with GPUs that match the resource block at the top. The
default predictor is Boltz via ``scripts/s4_boltz.sh``; switch to
``scripts/s4_alphafold.sh`` by overriding ``predict_script``.
"""

import asyncio
import os
from pathlib import Path

from radical.asyncflow import RadicalExecutionBackend  # type: ignore

from rome.flows.proteinbindingflow import ProteinBindingFlow
from rome.protein import BackboneSpec, ProteinBindingFlowConfig


HERE = Path(__file__).parent
STRUCTURES_DIR = HERE / "structures"
SCRIPTS_DIR = HERE / "scripts"

# Hardcoded peptide target — matches IMPRESS update_usecase/protein_binding
# (last 10 residues of Alpha Synuclein for the PDZ design problem).
TARGET_PEPTIDE = os.environ.get("ROME_TARGET_PEPTIDE", "EGYQDYEPEA")


def _discover_backbones() -> list[BackboneSpec]:
    """Pick up every .pdb under ./structures/ as a backbone."""
    specs = []
    for p in sorted(STRUCTURES_DIR.glob("*.pdb")):
        specs.append(
            BackboneSpec(
                backbone_id=p.stem,
                pdb_path=str(p),
                design_chains="A",
                target_peptide=TARGET_PEPTIDE,
                target_chain_name="pep",
                designed_chain_name="pdz",
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
        num_predict_workers=2,
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
        # Backends — point these at your local installs.
        mpnn_backend="foundry",
        mpnn_weights_dir=os.environ.get("MPNN_WEIGHTS_DIR"),
        mpnn_checkpoint_dir=os.environ.get("MPNN_CKPT_DIR"),
        # Default to Boltz; swap to scripts/s4_alphafold.sh for AF2.
        predict_script=os.environ.get(
            "PREDICT_SCRIPT", str(SCRIPTS_DIR / "s4_boltz.sh"),
        ),
        predict_cache_dir=os.environ.get("BOLTZ_CACHE"),
        extract_script=os.environ.get(
            "EXTRACT_SCRIPT", str(SCRIPTS_DIR / "s5_plddt_extract.sh"),
        ),
        base_path=os.environ.get("ROME_BASE_PATH", "./protein_binding_run"),
    )

    flow = ProteinBindingFlow(config=config, asyncflow=backend)
    await flow.launch()
    await backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
