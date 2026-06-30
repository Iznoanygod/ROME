"""IMPRESS protein-binding workflow with ROME attached as a shim.

This is the ``examples/protien_binding_usecase/run_protein_binding.py``
from the IMPRESS main branch, with three lines added to bring ROME's
corpus + continuous MPNN fine-tuning along for the ride. Nothing else
about the IMPRESS pipeline (``ProteinBindingPipeline``,
``ImpressManager``, the s1..s5 stages) changes.

The three additions are marked ``# ROME +``.
"""

import copy
import shutil
import asyncio
from typing import Any, Dict, Optional

from radical.asyncflow import RadicalExecutionBackend  # type: ignore

from impress import PipelineSetup, ImpressManager                       # type: ignore
from impress.pipelines.protein_binding import ProteinBindingPipeline    # type: ignore

# ROME + ------------------------------------------------------------------
from rome.impress import CorpusThresholds, RomeShim
# -------------------------------------------------------------------------


async def adaptive_criteria(current_score: float, previous_score: float) -> bool:
    """Returns True iff quality has degraded (the IMPRESS-paper convention
    that a 'higher' score is worse, e.g. inter-chain pAE)."""
    return current_score > previous_score


async def adaptive_decision(pipeline: ProteinBindingPipeline) -> Optional[Dict[str, Any]]:
    """Original IMPRESS adaptive_decision, verbatim from the main branch.

    Reads the per-pass CSV, identifies regressed proteins, spawns a child
    pipeline (capped at MAX_SUB_PIPELINES).
    """
    MAX_SUB_PIPELINES: int = 3
    sub_iter_seqs: Dict[str, str] = {}

    file_name = f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv"
    with open(file_name) as fd:
        for line in fd.readlines()[1:]:
            line = line.strip()
            if not line:
                continue
            name, *_, score_str = line.split(",")
            protein = name.split(".")[0]
            pipeline.current_scores[protein] = float(score_str)

    if not pipeline.previous_scores:
        pipeline.logger.pipeline_log("Saving current scores as previous and returning")
        pipeline.previous_scores = copy.deepcopy(pipeline.current_scores)
        return

    sub_iter_seqs = {}
    for protein, curr_score in pipeline.current_scores.items():
        if protein not in pipeline.iter_seqs:
            continue
        if await adaptive_criteria(curr_score, pipeline.previous_scores[protein]):
            sub_iter_seqs[protein] = pipeline.iter_seqs.pop(protein)

    if sub_iter_seqs and pipeline.sub_order < MAX_SUB_PIPELINES:
        new_name = f"{pipeline.name}_sub{pipeline.sub_order + 1}"
        pipeline.set_up_new_pipeline_dirs(new_name)
        for protein in sub_iter_seqs:
            src = f"{pipeline.output_path_af}/{protein}.pdb"
            dst = f"{pipeline.base_path}/{new_name}_in/{protein}.pdb"
            shutil.copyfile(src, dst)

        new_config = {
            "name": new_name,
            "type": type(pipeline),
            "adaptive_fn": adaptive_decision,
            "config": {
                "is_child": True,
                "start_pass": pipeline.passes,
                "passes": pipeline.passes,
                "iter_seqs": sub_iter_seqs,
                "seq_rank": pipeline.seq_rank + 1,
                "sub_order": pipeline.sub_order + 1,
                "previous_scores": copy.deepcopy(pipeline.previous_scores),
            },
        }
        pipeline.submit_child_pipeline_request(new_config)
        pipeline.finalize(sub_iter_seqs)
        if not pipeline.fasta_list_2:
            pipeline.kill_parent = True
    else:
        pipeline.previous_scores = copy.deepcopy(pipeline.current_scores)


async def impress_protein_bind() -> None:
    backend = await RadicalExecutionBackend(
        {
            "gpus": 1,
            "cores": 32,
            "runtime": 23 * 60,
            "resource": "purdue.anvil_gpu",
        }
    )

    manager = ImpressManager(execution_backend=backend)

    # ROME + ----------------------------------------------------------------
    # Score-gated corpus + continuous MPNN fine-tuning. Hooked into IMPRESS
    # by wrapping the adaptive_fn and entering shim.attached(manager) for
    # the duration of the run. No other IMPRESS code changes.
    shim = RomeShim(
        corpus_thresholds=CorpusThresholds(
            min_pLDDT=80.0, min_pTM=0.80, max_pAE=5.0,
        ),
        train_batch_threshold=64,
        train_shard_size=256,
        mpnn_checkpoint_dir="./mpnn_ckpts",
    )
    # -----------------------------------------------------------------------

    pipeline_setups = [
        PipelineSetup(
            name="p1",
            type=ProteinBindingPipeline,
            adaptive_fn=shim.wrap_adaptive_fn(adaptive_decision),   # ROME +
        )
    ]

    async with shim.attached(manager) as rome:                       # ROME +
        await manager.start(pipeline_setups=pipeline_setups)
        print(f"ROME: corpus={rome.corpus_size}, training rounds={rome.training_rounds}")

    await manager.flow.shutdown()


if __name__ == "__main__":
    asyncio.run(impress_protein_bind())
