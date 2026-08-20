"""IMPRESS-R: ROME-A attached to a real IMPRESS pipeline via ``adaptive_fn``.

Modelled directly on IMPRESS's protein-binding use case
(``examples/protien_binding_usecase/run_protein_binding.py`` on the
``archive/ipdps_pdz_usecase`` branch). The pipeline structure, the pass loop,
the score CSV and the degradation criterion are IMPRESS's; the AlphaFold and
ProteinMPNN executables are stubbed so the whole thing runs on a laptop.

The point is the seam. In the real use case ``adaptive_decision(pipeline)``
runs after the pLDDT-extraction task of every pass, reads
``af_stats_{name}_pass_{n}.csv``, and decides which designs regressed. That is
the natural — and only — place where a campaign both *has* fresh scored designs
and is *between* passes, so it is where both halves of ROME-A belong:

    contribute   rome.add_training_data(...)   for designs that pass the filter
    collect      rome.get_current_model()      and point MPNN at it next pass

Everything else in IMPRESS is untouched: ``run()`` never mentions ROME-A, and
the degradation logic that spawns child pipelines is IMPRESS's own, unchanged.

ROME-A builds its own workflow engine here rather than borrowing IMPRESS's, so
its training tasks are scheduled independently of the campaign's. Pass
``rome.Manager(impress_manager.flow)`` instead to share one engine — see
``docs/impress.md``.

Run::

    dragon -s examples/impress_r/adaptive_rome.py

Needs IMPRESS installed from the archived branch; see ``docs/impress.md`` for
the one backend rename it requires on current asyncflow.
"""

import asyncio
import csv
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from impress import ImpressBasePipeline, ImpressManager, PipelineSetup
from radical.asyncflow import LocalExecutionBackend

import rome
from rome.dummy import DummyTrainer

# The ProteinMPNN corpus helpers ship with this example (mpnn.py), not the
# framework — import whether run as a script here or as examples.impress_r.*.
try:
    from examples.impress_r.mpnn import impress_corpus_filter
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpnn import impress_corpus_filter

MAX_PASSES = 10
DESIGNS_PER_PASS = 8
#: IMPRESS's own confidence thresholds — the same ones that decide a design is
#: worth keeping decide whether it is worth training on.
MIN_PLDDT, MIN_PTM, MAX_PAE = 80.0, 0.80, 5.0


# ---------------------------------------------------------------------------
# The IMPRESS pipeline. Shaped like ProteinBindingPipeline, executables stubbed.
# ---------------------------------------------------------------------------

class ProteinBindingPipeline(ImpressBasePipeline):
    """MPNN -> AlphaFold -> pLDDT extraction, once per pass.

    Mirrors the real pipeline's attribute names (``passes``, ``iter_seqs``,
    ``current_scores``, ``previous_scores``, ``output_path_af``) so the adaptive
    function below is the same code the real use case would run.
    """

    def __init__(self, name: str, flow: Any, configs: Dict[str, Any] = None, **kwargs):
        configs = configs or {}
        self.passes: int = kwargs.get("passes", 1)
        self.max_passes: int = kwargs.get("max_passes", MAX_PASSES)
        self.sub_order: int = kwargs.get("sub_order", 0)
        self.iter_seqs: Dict[str, str] = kwargs.get("iter_seqs", {})
        self.current_scores: Dict[str, float] = {}
        self.previous_scores: Dict[str, float] = kwargs.get("previous_scores", {})
        self.base_path: str = kwargs.get("base_path", os.getcwd())
        self.output_path_af: str = os.path.join(self.base_path, f"{name}_af_out")
        #: Weights MPNN runs with. ROME-A republishes this between passes.
        self.mpnn_weights: str = kwargs.get("mpnn_weights", "proteinmpnn_v_48_020.pt")
        os.makedirs(self.output_path_af, exist_ok=True)
        super().__init__(name, flow, **configs, **kwargs)

    def register_pipeline_tasks(self) -> None:
        @self.auto_register_task()
        async def s1_mpnn(*args, **kwargs) -> str:
            """ProteinMPNN. The real task shells out to mpnn_wrapper.py."""
            return f"/bin/echo mpnn pass={self.passes} weights={self.mpnn_weights}"

        @self.auto_register_task()
        async def s4_alphafold(*args, **kwargs) -> str:
            """AlphaFold. The real task shells out to af2_multimer_reduced.sh."""
            return f"/bin/echo alphafold pass={self.passes}"

        # local_task=True keeps this a Python call rather than a shell command —
        # the seam IMPRESS provides for in-process steps.
        @self.auto_register_task(local_task=True)
        async def s5_extract() -> str:
            """Write af_stats_{name}_pass_{n}.csv, as IMPRESS's extractor does.

            Designs improve as the MPNN weights advance, which is what makes the
            closed loop visible in a run with no real models in it.
            """
            version = _version_of(self.mpnn_weights)
            boost = 4.0 * version
            path = os.path.join(
                self.base_path, f"af_stats_{self.name}_pass_{self.passes}.csv"
            )
            with open(path, "w", newline="") as fd:
                writer = csv.writer(fd)
                writer.writerow(["ID", "avg_plddt", "ptm", "avg_pae"])
                for i in range(DESIGNS_PER_PASS):
                    design = f"{self.name}_p{self.passes}_d{i}"
                    self.iter_seqs[design] = "".join(
                        random.choices("ACDEFGHIKLMNPQRSTVWY", k=60)
                    )
                    # The AF2 prediction of the designed sequence: coordinates
                    # and sequence in one file, which is the training example.
                    open(os.path.join(self.output_path_af, f"{design}.pdb"), "w").close()
                    writer.writerow([
                        f"{design}.pdb",
                        round(min(99.0, random.gauss(80.0 + boost, 5.0)), 2),
                        round(min(0.99, random.gauss(0.82 + boost / 100, 0.05)), 3),
                        round(max(0.5, random.gauss(4.5 - boost / 10, 1.0)), 2),
                    ])
            return path

    async def run(self) -> None:
        """IMPRESS's pass loop. Note that it never mentions ROME-A."""
        while self.passes <= self.max_passes:
            self.logger.pipeline_log(f"pass {self.passes} | mpnn={self.mpnn_weights}")
            await self.s1_mpnn()
            await self.s4_alphafold()
            await self.s5_extract()
            # Hands control to adaptive_fn and waits for it.
            await self.run_adaptive_step(wait=True)
            self.passes += 1

    async def finalize(self) -> None:
        shutil.rmtree(self.output_path_af, ignore_errors=True)


def _version_of(weights_path: str) -> int:
    """Read the round number out of a published checkpoint path."""
    for part in str(weights_path).split(os.sep):
        if part.startswith("v") and part[1:].isdigit():
            return int(part[1:])
    return 0


# ---------------------------------------------------------------------------
# The seam: ROME-A lives entirely inside adaptive_fn.
# ---------------------------------------------------------------------------

def make_adaptive_fn(manager: rome.Manager):
    """Build IMPRESS's adaptive function with ROME-A folded into it.

    ``manager`` is captured rather than passed through IMPRESS, because
    ``adaptive_fn`` has a fixed one-argument signature.
    """

    async def adaptive_decision(pipeline: ProteinBindingPipeline) -> None:
        stats = os.path.join(
            pipeline.base_path,
            f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv",
        )
        if not os.path.exists(stats):
            return

        # -- 1. contribute this pass's designs to ROME-A -------------------
        accepted = 0
        with open(stats) as fd:
            for row in csv.DictReader(fd):
                design = row["ID"].split(".")[0]
                uid = manager.add_training_data(
                    # The structure file IS the training example.
                    path=os.path.join(pipeline.output_path_af, f"{design}.pdb"),
                    sequence=pipeline.iter_seqs.get(design, ""),
                    backbone_id=pipeline.name,
                    pLDDT=float(row["avg_plddt"]),
                    pTM=float(row["ptm"]),
                    pAE=float(row["avg_pae"]),
                    score=float(row["avg_plddt"]),
                )
                accepted += uid is not None
                # IMPRESS's own criterion is the last column, avg_pae.
                pipeline.current_scores[design] = float(row["avg_pae"])

        # -- 2. collect an improved model for the next pass ----------------
        weights = manager.get_current_model()
        if weights and weights != pipeline.mpnn_weights:
            pipeline.mpnn_weights = weights
            pipeline.logger.pipeline_log(f"ROME-A published v{_version_of(weights)}")

        pipeline.logger.pipeline_log(
            f"corpus {manager.data.total_count} (+{accepted} this pass) | "
            f"{manager.get_training_status().name}"
        )

        # -- 3. IMPRESS's own degradation logic, unchanged -----------------
        if not pipeline.previous_scores:
            pipeline.previous_scores = dict(pipeline.current_scores)
            return
        # Higher pAE is worse; a regressed design would be migrated to a child
        # pipeline here in the real use case.
        pipeline.previous_scores = dict(pipeline.current_scores)

    return adaptive_decision


# ---------------------------------------------------------------------------

async def main() -> None:
    workdir = tempfile.mkdtemp(prefix="impress_r_")
    os.chdir(workdir)

    # ROME-A with no engine of its own passed in: it builds one at start() and
    # shuts it down at stop(), so its training tasks are managed independently
    # of IMPRESS's. Pass impress_manager.flow instead to share one.
    manager = rome.Manager(
        data_config=rome.DataConfig(
            # Only about a fifth of designs clear the confidence thresholds
            # before training starts, so the threshold counts accepted designs,
            # not designs attempted.
            min_samples=4,
            filter_func=impress_corpus_filter(MIN_PLDDT, MIN_PTM, MAX_PAE),
            dedup_key=lambda record: record["sequence"],
            sampling="top_k",
            shard_size=64,
            score_key="pLDDT",
        ),
        trainer_config=rome.TrainerConfig(
            # Swap for ProteinMPNNTrainer(ProteinMPNNConfig(...)) on a real run.
            trainer=DummyTrainer(train_seconds=0.5, gpus=1),
            checkpoint_dir=os.path.join(workdir, "checkpoints"),
            poll_interval=0.2,
        ),
    )
    await manager.start()

    impress_backend = await LocalExecutionBackend(ThreadPoolExecutor())
    impress = ImpressManager(execution_backend=impress_backend)

    try:
        await impress.start(pipeline_setups=[
            PipelineSetup(
                name="p1",
                type=ProteinBindingPipeline,
                adaptive_fn=make_adaptive_fn(manager),
            )
        ])
        print("\nROME-A:", manager.report())
    finally:
        await impress.flow.shutdown()
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
