"""IMPRESS's dummy adaptive example, hooked to ROME-A's dummy trainer.

This is IMPRESS's own ``examples/dummy_adaptive.py`` — the minimal adaptive
pipeline, ``sequence_analysis -> fitness_evaluation -> [adaptive step] ->
optimization_step``, with random child-pipeline spawning — with **two lines of
ROME-A** added inside the adaptive function:

    contribute   manager.add_training_data(...)   # this generation's designs
    collect      manager.get_current_model()      # the improved model, if any

Nothing else changes. ROME-A's training manager watches the corpus those
contributions build and, once ``min_samples`` designs have arrived from the
workflow, runs a training round on its own — here the ``DummyTrainer``, which
sleeps instead of fine-tuning and writes a checkpoint. The next generation to
call ``get_current_model`` picks it up. The pipeline code never schedules
training and never blocks on it.

To make the loop visible with no real model in it, ``fitness_evaluation``
produces better designs as the published model version climbs — so a child
generation running a freshly trained checkpoint scores higher than its parent.

Run::

    dragon -s examples/impress_r/dummy_adaptive_rome.py

Needs IMPRESS installed from the ``archive/ipdps_pdz_usecase`` branch; see
``docs/impress.md``. The one change from IMPRESS's example is the backend:
``ConcurrentExecutionBackend`` (asyncflow 0.2.0) is ``LocalExecutionBackend`` on
current asyncflow.
"""

import asyncio
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from impress import ImpressBasePipeline, ImpressManager, PipelineSetup
from radical.asyncflow import LocalExecutionBackend

import rome
from rome.dummy import DummyTrainer

#: Candidate sequences a generation's fitness step produces — the designs it
#: contributes to ROME-A. A small batch so a round fires after a pipeline or two.
DESIGNS_PER_GENERATION = 4


class DummyProteinPipeline(ImpressBasePipeline):
    """IMPRESS's dummy adaptive pipeline, plus a model it runs and a fitness.

    ``mpnn_weights`` is the only addition to IMPRESS's version: the model this
    generation runs, carried into child generations so an improved checkpoint
    propagates. Everything else — the three tasks, the ``run`` order, the
    adaptive step — is IMPRESS's.
    """

    def __init__(self, name: str, flow: Any, configs: Dict[str, Any] = None, **kwargs):
        # A child pipeline's ``config`` dict arrives spread as **kwargs (the
        # manager builds it as ``{**setup.config, **setup.kwargs}``), so read the
        # generation state from there, and pop it so it does not reach the base.
        merged = {**(configs or {}), **kwargs}
        self.iter_seqs: str = "MKFLVLACGT"
        self.generation: int = merged.pop("generation", 1)
        self.parent_name: str = merged.pop("parent_name", "root")
        self.max_generations: int = merged.pop("max_generations", 3)
        #: The model this generation runs. ROME-A republishes it between rounds.
        self.mpnn_weights: str = merged.pop("mpnn_weights", "proteinmpnn_v_48_020.pt")
        #: Designs and their fitness, produced by ``fitness_evaluation``.
        self.designs: Dict[str, float] = {}
        super().__init__(name, flow, **merged)

    def register_pipeline_tasks(self) -> None:
        @self.auto_register_task()
        async def sequence_analysis(*args, **kwargs) -> str:
            return "/bin/echo 'Analyzing' && /bin/date"

        # local_task=True: a Python step, so it can produce designs in-process.
        @self.auto_register_task(local_task=True)
        async def fitness_evaluation() -> Dict[str, float]:
            """Score this generation's designs.

            Fitness rises with the published model version, so a child running a
            freshly trained checkpoint beats its parent — which is the closed
            loop, made visible without a real model.
            """
            boost = 4.0 * _version_of(self.mpnn_weights)
            self.designs = {}
            for i in range(DESIGNS_PER_GENERATION):
                design = f"{self.name}_g{self.generation}_d{i}"
                self.iter_seqs = "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=60))
                self.designs[design] = round(random.gauss(70.0 + boost, 5.0), 2)
            return self.designs

        @self.auto_register_task()
        async def optimization_step(*args, **kwargs) -> str:
            return "/bin/echo 'Optimizing' && /bin/date"

    async def run(self) -> None:
        """IMPRESS's dummy pass. It never mentions ROME-A."""
        self.logger.pipeline_log(f"gen {self.generation} | mpnn={self.mpnn_weights}")
        await self.sequence_analysis()
        await self.fitness_evaluation()
        # Hands control to the adaptive function and waits for it.
        await self.run_adaptive_step(wait=True)
        await self.optimization_step()

    async def finalize(self) -> None:
        pass


def _version_of(weights_path: str) -> int:
    """Read the round number out of a published checkpoint path (``.../v3``)."""
    import os

    for part in str(weights_path).split(os.sep):
        if part.startswith("v") and part[1:].isdigit():
            return int(part[1:])
    return 0


def make_adaptive_fn(manager: rome.Manager):
    """IMPRESS's adaptive strategy with the two ROME-A hooks folded in.

    ``manager`` is captured because ``adaptive_fn`` has a fixed one-argument
    signature — the manager cannot be threaded through IMPRESS.
    """

    async def adaptive_optimization_strategy(pipeline: DummyProteinPipeline) -> None:
        # -- HOOK 1: contribute this generation's designs to ROME-A ----------
        accepted = 0
        for design, fitness in pipeline.designs.items():
            uid = manager.add_training_data(
                sequence=pipeline.iter_seqs,
                score=fitness,
                backbone_id=pipeline.name,
            )
            accepted += uid is not None

        # -- HOOK 2: collect an improved model for the next generation -------
        weights = manager.get_current_model()
        if weights and weights != pipeline.mpnn_weights:
            pipeline.mpnn_weights = weights
            pipeline.logger.pipeline_log(f"ROME-A published v{_version_of(weights)}")

        pipeline.logger.pipeline_log(
            f"corpus {manager.data.total_count} (+{accepted} this gen) | "
            f"{manager.get_training_status().name}"
        )

        # -- IMPRESS's own child-spawn logic, unchanged but for carrying the
        #    current model into the child so the improvement propagates. -------
        if pipeline.generation >= pipeline.max_generations or random.random() >= 0.5:
            return
        pipeline.submit_child_pipeline_request({
            "name": f"{pipeline.name}_g{pipeline.generation + 1}",
            "type": type(pipeline),
            "config": {
                "generation": pipeline.generation + 1,
                "parent_name": pipeline.name,
                "max_generations": pipeline.max_generations,
                "mpnn_weights": pipeline.mpnn_weights,
            },
            "adaptive_fn": make_adaptive_fn(manager),
        })

    return adaptive_optimization_strategy


async def main() -> None:
    workdir = tempfile.mkdtemp(prefix="impress_r_dummy_")

    # ROME-A with no engine passed in: it builds its own at start() and shuts it
    # down at stop(), so its training runs independently of IMPRESS's tasks.
    manager = rome.Manager(
        data_config=rome.DataConfig(
            # Enough designs from the workflow to make a round worthwhile. A
            # generation contributes DESIGNS_PER_GENERATION, so a round fires
            # after the first pipeline or two.
            min_samples=8,
        ),
        trainer_config=rome.TrainerConfig(
            # Swap for ProteinMPNNTrainer(ProteinMPNNConfig(...)) on a real run.
            trainer=DummyTrainer(train_seconds=0.5, gpus=1),
            checkpoint_dir=f"{workdir}/checkpoints",
            poll_interval=0.2,
        ),
    )
    await manager.start()

    impress_backend = await LocalExecutionBackend(ThreadPoolExecutor())
    impress = ImpressManager(impress_backend)

    try:
        await impress.start(pipeline_setups=[
            PipelineSetup(
                name=f"p{i}",
                type=DummyProteinPipeline,
                adaptive_fn=make_adaptive_fn(manager),
            )
            for i in range(1, 4)
        ])
        print("\nROME-A:", manager.report())
    finally:
        await impress.flow.shutdown()
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
