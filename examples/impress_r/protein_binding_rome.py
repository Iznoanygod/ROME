"""IMPRESS-R: the real protein-binding use case with ROME-A folded in.

This is IMPRESS's ``examples/protien_binding_usecase/run_protein_binding.py`` —
the real ``ProteinBindingPipeline`` (MPNN -> AlphaFold -> pLDDT extraction, the
pass loop, the score CSV, the degrade-and-migrate criterion) — with ROME-A added
inside ``adaptive_decision`` and nowhere else:

    contribute   manager.add_training_data(...)   this pass's scored design(s)
    collect      manager.get_current_model()      confirm the improved model

Once ``min_samples`` designs have accumulated from the campaign, ROME-A's
training manager fine-tunes ProteinMPNN on its own and, with
``publish_into_repo``, writes the new weights into the ProteinMPNN checkout's
``vanilla_model_weights/`` — the file ``protein_mpnn_run.py`` loads by default.
So the campaign's next MPNN pass runs the improved model with no change to the
wrapper. IMPRESS's ``run()`` never mentions ROME-A.

**Running on Delta.** This needs the same environment the campaign does — the
ProteinMPNN checkout, PyRosetta, AlphaFold, and the three usecase scripts
(``mpnn_wrapper.py``, ``af2_multimer_reduced.sh``, ``plddt_extract_pipeline.py``)
plus the ``{name}_in/`` inputs in the working directory. Run it from the usecase
directory:

    cd .../IMPRESS/examples/protien_binding_usecase
    ROME_MPNN_REPO=$WORK/ProteinMPNN \
      dragon -s .../ROME/examples/impress_r/protein_binding_rome.py

Environment:

* ``ROME_MPNN_REPO`` — the dauparas/ProteinMPNN checkout IMPRESS runs (the ``-mpnn``
  path). Required for a real training round.
* ``ROME_TRAINER`` — ``mpnn`` (default) fine-tunes ProteinMPNN; ``dummy`` sleeps
  and writes a placeholder, for a wiring smoke test with no torch or GPU.
* ``ROME_IMPRESS_BACKEND`` — ``local`` (default; task subprocesses on this node,
  right for a single-node interactive job) or ``dragon`` (multi-process/node).
  The upstream script uses ``RadicalExecutionBackend``, which exists only in
  asyncflow 0.2.0; on current asyncflow use one of these. See ``docs/impress.md``.

The two open items from ``docs/impress.md`` apply to a production run: the
prediction path is overwritten each pass (handled here — the hook copies each
prediction aside before recording it), and the corpus filter must be calibrated
on your own AF2 scores (``percentile_sampler`` needs no calibration and is used
below).
"""

import asyncio
import copy
import csv
import os
import shutil
import tempfile
from typing import Any, Dict, Optional

from impress import ImpressManager, PipelineSetup
from impress.pipelines.protein_binding import ProteinBindingPipeline

import rome
from rome.train.mpnn import percentile_sampler

MPNN_REPO = os.environ.get(
    "ROME_MPNN_REPO", f"/anvil/scratch/{os.environ.get('USER', 'user')}/impress/ProteinMPNN"
)


class ProteinBindingPipelineR(ProteinBindingPipeline):
    """The real pipeline, with best-model selection made backend-agnostic.

    IMPRESS selects AlphaFold's best model in the AF task's ``post_exec`` —
    copying ``dimer_models/{target}/*ranked_0*.pdb`` into ``best_models/`` and
    the ranking JSON into ``best_ptm/``. ``post_exec`` is a RADICAL-Pilot
    feature; on LocalExecutionBackend or the Dragon backend it is ignored, so
    ``best_models`` stays empty and ``plddt_extract_pipeline.py`` writes a
    header-only CSV. (That is the empty-``af_stats`` failure — see
    ``scripts/populate_best_models.py`` to recover an already-run campaign.)

    This folds those copies into the AF task's own shell command with ``&&``, so
    they run on the same node on whatever backend, right after AlphaFold — no
    ``post_exec`` needed.
    """

    def register_pipeline_tasks(self) -> None:
        super().register_pipeline_tasks()          # registers s1..s5 as-is

        @self.auto_register_task()
        async def s4(target_fasta, task_description={"gpus_per_rank": 1}):  # noqa: B006
            pred = os.path.join(self.output_path, "af", "prediction")
            models = os.path.join(pred, "dimer_models", target_fasta)
            best_pdb = os.path.join(pred, "best_models", f"{target_fasta}.pdb")
            best_json = os.path.join(pred, "best_ptm", f"{target_fasta}.json")
            mpnn_pdb = os.path.join(self.output_path, "mpnn",
                                    f"job_{self.passes}", f"{target_fasta}.pdb")
            return (
                f"/bin/bash {self.base_path}/af2_multimer_reduced.sh "
                f"{self.output_path}/af/fasta/ {target_fasta}.fa "
                f"{pred}/dimer_models/ "
                # best-model selection, inline (post_exec's job, backend-agnostic)
                f"&& cp {models}/*ranked_0*.pdb {best_pdb} "
                f"&& cp {models}/*ranking_debug*.json {best_json} "
                f"&& cp {models}/*ranked_0*.pdb {mpnn_pdb}"
            )


# ---------------------------------------------------------------------------
# IMPRESS's migration criterion — verbatim from run_protein_binding.py.
# ---------------------------------------------------------------------------

async def adaptive_criteria(current_score: float, previous_score: float) -> bool:
    """Quality degraded (a binder's interface pAE rose) → migrate it."""
    return current_score > previous_score


# ---------------------------------------------------------------------------
# The seam: ROME-A lives entirely inside adaptive_decision.
# ---------------------------------------------------------------------------

def make_adaptive_decision(manager: rome.Manager, stage_dir: str):
    """IMPRESS's ``adaptive_decision`` with the two ROME-A hooks folded in.

    ``manager`` and ``stage_dir`` are captured because ``adaptive_fn`` has a
    fixed one-argument signature. Everything below the hooks is IMPRESS's own
    migration logic, unchanged.
    """
    os.makedirs(stage_dir, exist_ok=True)

    async def adaptive_decision(pipeline: ProteinBindingPipeline) -> Optional[Dict[str, Any]]:
        MAX_SUB_PIPELINES: int = 3
        sub_iter_seqs: Dict[str, str] = {}

        stats = f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv"
        accepted = 0
        with open(stats) as fd:
            for row in csv.DictReader(fd):          # ID, avg_plddt, ptm, avg_pae
                protein = row["ID"].split(".")[0]
                # IMPRESS's criterion is the interface pAE (last column).
                pipeline.current_scores[protein] = float(row["avg_pae"])

                # -- HOOK 1: contribute this design to ROME-A -----------------
                # The prediction is keyed by pipeline, not by pass, and is
                # deleted on migration, so copy it aside before recording it —
                # otherwise the corpus points at a file whose contents change.
                src = os.path.join(pipeline.output_path_af, f"{protein}.pdb")
                if not os.path.exists(src):
                    continue
                staged = os.path.join(
                    stage_dir, f"{pipeline.name}_pass{pipeline.passes}_{protein}.pdb"
                )
                shutil.copyfile(src, staged)
                ranked = pipeline.iter_seqs.get(protein) or []
                sequence = ranked[pipeline.seq_rank][0] if len(ranked) > pipeline.seq_rank else ""
                uid = manager.add_training_data(
                    path=staged,
                    sequence=sequence,
                    backbone_id=protein,               # cluster on the target
                    pLDDT=float(row["avg_plddt"]),
                    pTM=float(row["ptm"]),
                    pAE=float(row["avg_pae"]),
                    score=float(row["avg_plddt"]),
                )
                accepted += uid is not None

        # -- HOOK 2: collect the improved model ---------------------------------
        # The trainer publishes into the ProteinMPNN checkout's weights dir, so
        # the next MPNN pass picks it up with no wrapper change; this just reports
        # what ROME-A currently has.
        weights = manager.get_current_model()
        pipeline.logger.pipeline_log(
            f"ROME-A: corpus {manager.data.total_count} (+{accepted} this pass) | "
            f"{manager.get_training_status().name}"
            + (f" | model {os.path.basename(weights)}" if weights else "")
        )

        # ---------------------------------------------------------------------
        # IMPRESS's own migration logic below — verbatim.
        # ---------------------------------------------------------------------

        if not pipeline.previous_scores:
            pipeline.logger.pipeline_log("Saving current scores as previous and returning")
            pipeline.previous_scores = copy.deepcopy(pipeline.current_scores)
            return

        sub_iter_seqs = {}
        for protein, curr_score in pipeline.current_scores.items():
            if protein not in pipeline.iter_seqs:
                continue
            decision = await adaptive_criteria(curr_score, pipeline.previous_scores[protein])
            pipeline.logger.pipeline_log(f"Adaptive descision: {decision}")
            if decision:
                sub_iter_seqs[protein] = pipeline.iter_seqs.pop(protein)

        if sub_iter_seqs and pipeline.sub_order < MAX_SUB_PIPELINES:
            new_name: str = f"{pipeline.name}_sub{pipeline.sub_order + 1}"
            pipeline.set_up_new_pipeline_dirs(new_name)

            for protein in sub_iter_seqs:
                src = f"{pipeline.output_path_af}/{protein}.pdb"
                dst = f"{pipeline.base_path}/{new_name}_in/{protein}.pdb"
                shutil.copyfile(src, dst)

            new_config = {
                "name": new_name,
                "type": type(pipeline),
                "adaptive_fn": make_adaptive_decision(manager, stage_dir),
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

    return adaptive_decision


# ---------------------------------------------------------------------------

def _build_trainer(checkpoint_dir: str):
    """The ProteinMPNN trainer, or a dummy for a wiring smoke test."""
    if os.environ.get("ROME_TRAINER", "mpnn").lower() == "dummy":
        from rome.dummy import DummyTrainer

        return DummyTrainer(train_seconds=1.0, gpus=0)

    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    return ProteinMPNNTrainer(ProteinMPNNConfig(
        mpnn_repo=MPNN_REPO,
        # Fine-tune the weights IMPRESS runs and publish back into the repo so
        # the next pass picks them up. A binder is a dimer: chain A designed,
        # chain B (the peptide) is context — the trainer's defaults.
        publish_into_repo=True,
        model_name="v_48_020",
    ), gpus=1)


async def impress_protein_bind() -> None:
    workdir = tempfile.mkdtemp(prefix="impress_r_")

    # ROME-A with no engine passed in: it builds its own at start() and shuts it
    # down at stop(), so its training runs independently of IMPRESS's tasks.
    manager = rome.Manager(
        data_config=rome.DataConfig(
            # A campaign contributes ~one scored design per pipeline per pass,
            # so the corpus grows slowly; set this to how many accepted designs
            # a round should wait for. percentile_sampler needs no score
            # calibration — see docs/impress.md.
            min_samples=int(os.environ.get("ROME_MIN_SAMPLES", 8)),
            sample_func=percentile_sampler(0.33),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=_build_trainer(os.path.join(workdir, "checkpoints")),
            checkpoint_dir=os.path.join(workdir, "checkpoints"),
            poll_interval=1.0,
            # On the Dragon backend a finished round's result can be delivered
            # late while a task is running; publish from disk after this grace.
            result_fallback_seconds=float(os.environ.get("ROME_FALLBACK", 60)),
        ),
    )
    await manager.start()

    backend = await _build_impress_backend()
    impress = ImpressManager(execution_backend=backend)

    try:
        await impress.start(pipeline_setups=[
            PipelineSetup(
                name="p1",
                type=ProteinBindingPipelineR,
                adaptive_fn=make_adaptive_decision(manager, os.path.join(workdir, "designs")),
            )
        ])
        print("\nROME-A:", manager.report())
    finally:
        await impress.flow.shutdown()
        await manager.stop()


async def _build_impress_backend():
    """IMPRESS's execution backend. Local by default; Dragon on request.

    The upstream script uses RadicalExecutionBackend (asyncflow 0.2.0); on
    current asyncflow use one of these.
    """
    if os.environ.get("ROME_IMPRESS_BACKEND", "local").lower() == "dragon":
        from rhapsody.backends import DragonExecutionBackendV3

        return await DragonExecutionBackendV3({"results_ddict_mem": 2 * 1024 ** 3})
    from concurrent.futures import ThreadPoolExecutor

    from radical.asyncflow import LocalExecutionBackend

    return await LocalExecutionBackend(ThreadPoolExecutor())


if __name__ == "__main__":
    asyncio.run(impress_protein_bind())
