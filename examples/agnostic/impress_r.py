"""IMPRESS-R: adding model improvement to the IMPRESS pipeline with ROME-A.

IMPRESS today runs backbone -> ProteinMPNN -> structure prediction ->
pLDDT/pTM/pAE -> keep/fallback/migrate/drop. It is open loop: every campaign
improves the designs, never the model.

This example is what closing that loop costs. The IMPRESS pipeline below is a
stand-in (``run_impress_cycle``), and it runs *unchanged* — ROME-A is four
calls:

    1. build a Manager with a data policy and a trainer          (setup)
    2. rome.add_training_data(...) as designs are scored          (in the loop)
    3. rome.get_current_model() to pick up improved weights       (in the loop)
    4. await rome.stop()                                          (teardown)

ROME-A keeps its shared state in a Dragon DDict, so the example runs under the
Dragon runtime with a single-node execution backend::

    dragon examples/agnostic/impress_r.py

On a real allocation, swap ``LocalExecutionBackend`` for the Dragon or RADICAL
backend the campaign already uses — ROME-A submits its tasks to whatever engine
you hand it.
"""

import asyncio
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import rome
from rome.train.mpnn import (
    ProteinMPNNConfig,
    ProteinMPNNTrainer,
    impress_corpus_filter,
)

NUM_CYCLES = 8
DESIGNS_PER_CYCLE = 8


# ---------------------------------------------------------------------------
# The host workflow. ROME-A does not know or care what is in here.
# ---------------------------------------------------------------------------

def run_impress_cycle(cycle, mpnn_weights):
    """Stand-in for backbone -> MPNN -> predict -> extract metrics.

    Returns the per-design metrics a real campaign would parse out of
    ``plddt_extract_pipeline.py``. The fake "improvement" from a trained
    checkpoint is only here so the example prints something interesting.
    """
    trained_rounds = mpnn_weights.count("/v") if mpnn_weights else 0
    boost = 4.0 * trained_rounds

    designs = []
    for i in range(DESIGNS_PER_CYCLE):
        designs.append({
            "backbone_id": f"bb{i % 3}",
            "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=40)),
            "pdb_path": f"/scratch/cycle{cycle}/design{i}.pdb",
            "pLDDT": min(99.0, random.gauss(76.0 + boost, 6.0)),
            "pTM": min(0.99, random.gauss(0.78 + boost / 100, 0.06)),
            "pAE": max(0.5, random.gauss(6.0 - boost / 10, 1.5)),
        })
    return designs


def train_proteinmpnn(shard_path, output_dir, config):
    """Stand-in for foundry's MPNNTrainer.

    A real deployment drops ``backend='custom'`` and lets
    :class:`ProteinMPNNTrainer` call ``mpnn.trainers.mpnn.MPNNTrainer``
    directly; this keeps the example runnable without foundry installed.
    """
    with open(os.path.join(output_dir, "checkpoint.txt"), "w") as fd:
        fd.write(f"trained on {shard_path}\n")
    return output_dir


# ---------------------------------------------------------------------------
# ROME-A adoption
# ---------------------------------------------------------------------------

async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)
    checkpoint_dir = tempfile.mkdtemp(prefix="impress_r_")

    # (1) Setup. The data policy is IMPRESS's own confidence thresholds: only
    # designs the campaign is already confident in are worth training on.
    manager = rome.Manager(
        flow,
        data_config=rome.DataConfig(
            min_samples=16,
            filter_func=impress_corpus_filter(min_pLDDT=80.0, min_pTM=0.8, max_pAE=5.0),
            dedup_key=lambda record: record["sequence"],
            sampling="top_k",
            shard_size=64,
            score_key="pLDDT",
        ),
        trainer_config=rome.TrainerConfig(
            trainer=ProteinMPNNTrainer(
                ProteinMPNNConfig(
                    backend="custom",
                    train_func=train_proteinmpnn,
                    shard_format="jsonl",
                ),
                gpus=1,
            ),
            checkpoint_dir=checkpoint_dir,
            poll_interval=0.5,
        ),
    )
    await manager.start()

    try:
        for cycle in range(NUM_CYCLES):
            # (3) The pipeline picks up improved weights when they exist. Before
            # the first round this is None and IMPRESS runs exactly as it always
            # has.
            weights = manager.get_current_model()

            designs = run_impress_cycle(cycle, weights)

            # (2) Scored outputs go into the corpus. This is the only line the
            # host workflow adds to its inner loop.
            for design in designs:
                manager.add_training_data(score=design["pLDDT"], **design)

            mean_plddt = sum(d["pLDDT"] for d in designs) / len(designs)
            print(
                f"cycle {cycle}: mean pLDDT {mean_plddt:5.1f} | "
                f"corpus {manager.data.total_count:3d} "
                f"({manager.data.unconsumed_count} fresh) | "
                f"model v{manager.model_version} | "
                f"{manager.get_training_status().name}"
            )

            # Give the training manager room to notice the threshold and run a
            # round. A real campaign spends this time folding structures.
            await asyncio.sleep(1.0)

        print("\nfinal report:", manager.report())
    finally:
        # (4) Teardown. An in-flight round is waited out, not killed.
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
