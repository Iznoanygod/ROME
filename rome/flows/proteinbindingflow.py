"""ProteinBindingFlow — streaming MPNN + IMPRESS adaptive + continuous training.

Replaces nothing in ``streamflow.py``; lives alongside it. Does not import
torch/transformers/peft/rose: the LLM-specific flows can move to an optional
subpackage without affecting this one.

Pipeline shape (see design discussion in commit log / docs):

* N streaming ``mpnn_generate_loop`` workers continuously sample
  ProteinMPNN under the current weights, writing into per-backbone
  buffers in ``workflow_ddict["mpnn_outputs"]``.

* The :class:`LogLikelihoodRanker` sorts those into
  ``ranked_candidates[backbone_id]`` and the AF2 scheduler pulls the head.

* For each cycle, the top L1 candidate per backbone is run through
  ``af2_predict_task`` then ``extract_metrics_task``; metrics flow into
  ``cycle_results``.

* The L2 :class:`AdaptiveCriterion` decides keep / fallback / migrate /
  drop per backbone. Migrations build sub-pipelines via the
  :class:`AdaptiveCoordinator`.

* The :class:`CorpusCurator` writes qualifying (backbone, sequence, metrics)
  tuples to ``workflow_ddict["corpus"]`` and fires
  ``mpnn_train_task`` whenever ``train_batch_threshold`` new entries
  accumulate. On training completion, the orchestrator bumps
  ``model_version`` and the generators reload weights between batches.
"""

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

from rome.protein.config import ProteinBindingFlowConfig
from rome.protein.coordinator import AdaptiveCoordinator
from rome.protein.corpus import CorpusCurator
from rome.protein.criteria import (
    AdaptiveCriterion,
    CriterionInput,
    Decision,
    default_criterion,
)
from rome.protein.hooks import TaskHooks
from rome.protein.pipeline import ProteinBindingPipeline
from rome.protein.ranker import LogLikelihoodRanker
from rome.protein.schema import BackboneSpec, PredictionResult


class ProteinBindingFlow:
    """Orchestrator for the streaming-MPNN + adaptive-AF2 + continuous-training loop.

    Standalone — does not subclass ``rome.workflow.Workflow`` to avoid
    pulling its LLM-shaped signature in. When the LLM bits move to an
    optional subpackage, a non-LLM ``Workflow`` base can be introduced and
    this class can adopt it without changing behavior.
    """

    def __init__(
        self,
        *,
        config: ProteinBindingFlowConfig,
        asyncflow: Any = None,
        criterion: Optional[AdaptiveCriterion] = None,
        task_hooks: Optional[TaskHooks] = None,
        state_factory: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        config : ProteinBindingFlowConfig
            Pipeline knobs + science-tool paths.
        asyncflow : Any, optional
            RADICAL asyncflow execution backend. Optional so tests can run
            the orchestration without a real backend.
        criterion : AdaptiveCriterion, optional
            Override the L2 adaptive decision function. Defaults to the
            paper's policy.
        task_hooks : TaskHooks, optional
            Inject alternative implementations of the four science tools
            (MPNN generator, AF2 predict, extract metrics, MPNN train).
            Unset hooks fall through to :mod:`rome.protein.tasks`.
        state_factory : Callable[[], (ddict, event)], optional
            Override the shared-state factory. Defaults to Dragon's DDict +
            Event; tests can pass an in-memory stub.
        """
        self.config = config
        self.asyncflow = asyncflow
        self.criterion: AdaptiveCriterion = (
            criterion or config.adaptive_criterion or default_criterion
        )
        self.hooks: TaskHooks = (task_hooks or TaskHooks()).resolved()
        self._state_factory = state_factory or _new_shared_state
        self.coordinator = AdaptiveCoordinator()
        self.ranker = LogLikelihoodRanker()
        # populated in launch()
        self._workflow_ddict: Any = None
        self._terminate_event: Any = None
        self._generator_tasks: List[Any] = []
        self._background: Optional[asyncio.Future] = None
        self._corpus: Optional[CorpusCurator] = None
        self._train_tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # entrypoint
    # ------------------------------------------------------------------

    async def launch(self) -> None:
        """Run until every pipeline has drained."""
        self._workflow_ddict, self._terminate_event = self._state_factory()
        self._seed_state()

        # background coroutines (ranker, corpus curator)
        self._corpus = CorpusCurator(
            config=self.config,
            schedule_train_fn=self._schedule_training_round,
            on_train_scheduled=self._train_tasks.append,
        )
        self._background = asyncio.gather(
            self.ranker.run(self._workflow_ddict, self._terminate_event),
            self._corpus.run(self._workflow_ddict, self._terminate_event),
        )

        # streaming MPNN generator workers (hook-injected)
        for i in range(self.config.num_mpnn_generators):
            self._generator_tasks.append(
                asyncio.create_task(
                    self.hooks.mpnn_generator_loop(
                        self.config,
                        i,
                        self._workflow_ddict,
                        self._terminate_event,
                    )
                )
            )

        # seed root pipeline(s) — one per backbone if separate-pipelines mode,
        # else a single pipeline owning all backbones
        for pipeline in self._build_root_pipelines():
            await self.coordinator.submit(pipeline)

        # main loop: pull pipelines off the coordinator, run their cycle DAG,
        # process adaptive decisions, recurse on children
        await self._run_until_drained()

        # Force a final curator sweep so the background loop's poll interval
        # doesn't race with shutdown on fast runs (tests, small workloads).
        self._corpus.sweep(self._workflow_ddict)

        # let any in-flight training tasks complete before shutting down
        if self._train_tasks:
            await asyncio.gather(*self._train_tasks, return_exceptions=True)

        # shutdown
        self._terminate_event.set()
        for t in self._generator_tasks:
            t.cancel()
        try:
            await self._background
        except asyncio.CancelledError:
            pass
        # await cancelled generator tasks so the loop is clean
        for t in self._generator_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _seed_state(self) -> None:
        d = self._workflow_ddict
        d["backbone_pool"] = {s.backbone_id: s for s in self.config.structures}
        d["mpnn_outputs"] = {s.backbone_id: [] for s in self.config.structures}
        d["ranked_candidates"] = {s.backbone_id: [] for s in self.config.structures}
        d["cycle_results"] = {s.backbone_id: [] for s in self.config.structures}
        d["corpus"] = {}
        d["model_version"] = 0
        d["mpnn_checkpoint_path"] = self.config.mpnn_weights_dir
        d["train_in_flight"] = False
        d["global_cycle"] = 0

    def _build_root_pipelines(self) -> List[ProteinBindingPipeline]:
        os.makedirs(self.config.base_path, exist_ok=True)
        # default: single pipeline owning all backbones (IMPRESS's
        # "Single Pipeline with Parallel Structures" mode).
        backbones = {s.backbone_id: s for s in self.config.structures}
        root = ProteinBindingPipeline(
            pipeline_id=f"p_root_{uuid.uuid4().hex[:6]}",
            base_path=self.config.base_path,
            backbones=backbones,
            iter_seqs=dict(backbones),
        )
        root.set_up_dirs()
        return [root]

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    async def _run_until_drained(self) -> None:
        """Drain coordinator + handle child pipelines spawned mid-run.

        Pulls every pipeline off the submit queue and runs it as a task;
        new children submitted during a parent's adaptive step join the
        pool on the next sweep. Exits when no tasks are running and the
        queue is empty.
        """
        pending: set = set()
        while True:
            # Drain newly-submitted pipelines into the running set.
            while not self.coordinator._submit_queue.empty():
                pipeline = await self.coordinator.next_to_run()
                pending.add(asyncio.create_task(self._run_pipeline(pipeline)))
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending,
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Surface unexpected failures rather than swallowing them.
            for t in done:
                exc = t.exception()
                if exc is not None:
                    raise exc

    async def _run_pipeline(self, pipeline: ProteinBindingPipeline) -> None:
        """Run cycles for one pipeline until it terminates or migrates out."""
        try:
            while pipeline.iter_seqs and pipeline.passes < self.config.max_cycles:
                self._workflow_ddict["global_cycle"] = pipeline.passes
                await self._run_one_cycle(pipeline)
                pipeline.passes += 1
                if pipeline.kill_parent:
                    break
        finally:
            await self.coordinator.mark_complete(pipeline.pipeline_id)

    async def _run_one_cycle(self, pipeline: ProteinBindingPipeline) -> None:
        """One cycle = each backbone reaches a non-FALLBACK decision once.

        Per IMPRESS §II-C Stage 6: a regression triggers AF2 re-run with
        the next-best sequence (up to ``max_fallback_sequences``) *within
        the same cycle*. The cycle only advances when every backbone
        either KEEPs, MIGRATEs, or DROPs.
        """
        backbones_to_resolve = list(pipeline.iter_seqs.keys())
        cycle_metrics: List[PredictionResult] = []
        migrating: List[str] = []

        # Reset fallback counters at cycle start.
        for bid in backbones_to_resolve:
            pipeline.fallback_attempts[bid] = 0

        # Resolve each backbone in parallel.
        results = await asyncio.gather(
            *(self._resolve_backbone(pipeline, bid) for bid in backbones_to_resolve),
            return_exceptions=True,
        )
        for bid, outcome in zip(backbones_to_resolve, results):
            if isinstance(outcome, Exception):
                continue
            decision, af2 = outcome
            if af2 is not None:
                cycle_metrics.append(af2)
            if decision == Decision.MIGRATE:
                migrating.append(bid)
            elif decision == Decision.DROP:
                pipeline.iter_seqs.pop(bid, None)

        # Record this cycle's final metrics + update score frontier.
        self._record_cycle(pipeline, cycle_metrics)

        # Migrate any backbones flagged during the resolve step.
        if migrating:
            moved = pipeline.migrate_backbones(migrating)
            child = self.coordinator.submit_child_pipeline_request(pipeline, moved)
            await self.coordinator.submit(child)
            if not pipeline.iter_seqs:
                pipeline.kill_parent = True

    async def _resolve_backbone(
        self, pipeline: ProteinBindingPipeline, backbone_id: str,
    ):
        """AF2 + criterion in a retry loop until a non-FALLBACK decision.

        Returns ``(Decision, PredictionResult | None)``. On KEEP/MIGRATE the
        consumed L1 candidate (plus any fallback losers ahead of it) is
        popped from ``ranked_candidates``; on DROP the bucket is left
        alone since the backbone is no longer tracked.
        """
        last_af2: Optional[PredictionResult] = None
        last_decision: Optional[Decision] = None
        while True:
            record = await self._next_l1_candidate(pipeline, backbone_id)
            if record is None:
                return Decision.DROP, last_af2

            af2_results = await self._predict_and_extract(
                pipeline, backbone_id, record
            )
            if not af2_results:
                # No metrics: treat as fallback unless we're out of budget.
                pipeline.fallback_attempts[backbone_id] = (
                    pipeline.fallback_attempts.get(backbone_id, 0) + 1
                )
                if pipeline.fallback_attempts[backbone_id] >= (
                    self.config.max_fallback_sequences
                ):
                    return Decision.DROP, last_af2
                continue

            af2 = af2_results[0]
            last_af2 = af2
            pipeline.current_scores[backbone_id] = {
                "pLDDT": af2.pLDDT,
                "pTM": af2.pTM,
                "pAE": af2.pAE,
            }
            inp = CriterionInput(
                backbone_id=backbone_id,
                current=pipeline.current_scores.get(backbone_id, {}),
                previous=pipeline.previous_scores.get(backbone_id) or None,
                fallback_attempts=pipeline.fallback_attempts.get(backbone_id, 0),
                sub_order=pipeline.sub_order,
                config=self.config,
            )
            last_decision = await self.criterion(inp)

            if last_decision == Decision.FALLBACK:
                pipeline.fallback_attempts[backbone_id] = (
                    pipeline.fallback_attempts.get(backbone_id, 0) + 1
                )
                continue  # retry within this cycle

            if last_decision == Decision.KEEP:
                pipeline.previous_scores[backbone_id] = dict(
                    pipeline.current_scores[backbone_id]
                )
                self._consume_l1_head(backbone_id, pipeline)
                return Decision.KEEP, af2

            if last_decision == Decision.MIGRATE:
                self._consume_l1_head(backbone_id, pipeline)
                return Decision.MIGRATE, af2

            if last_decision == Decision.DROP:
                return Decision.DROP, af2

    async def _next_l1_candidate(
        self, pipeline: ProteinBindingPipeline, backbone_id: str,
    ) -> Optional[dict]:
        """Block until a fresh L1 candidate is available for ``backbone_id``.

        Returns ``None`` if termination is signaled before one appears.
        """
        offset = pipeline.fallback_attempts.get(backbone_id, 0)
        while not self._terminate_event.is_set():
            ranked = self._workflow_ddict.get("ranked_candidates", {}) or {}
            bucket = ranked.get(backbone_id, [])
            if offset < len(bucket):
                return bucket[offset]
            await asyncio.sleep(0.05)
        return None

    def _consume_l1_head(
        self, backbone_id: str, pipeline: ProteinBindingPipeline,
    ) -> None:
        """Pop the consumed candidate (+ any fallback losers ahead of it)."""
        ranked = self._workflow_ddict.get("ranked_candidates", {}) or {}
        bucket = ranked.get(backbone_id, [])
        offset = pipeline.fallback_attempts.get(backbone_id, 0)
        # KEEP/MIGRATE consume bucket[0..offset+1]: all fallback losers + winner
        ranked[backbone_id] = bucket[offset + 1:]
        self._workflow_ddict["ranked_candidates"] = ranked

    async def _predict_and_extract(
        self,
        pipeline: ProteinBindingPipeline,
        backbone_id: str,
        record: dict,
    ) -> List[PredictionResult]:
        """Run s3 -> s4 -> s4_post_exec -> s5 for a single L1 candidate.

        Mirrors the IMPRESS update_usecase/protein_binding stage ordering:
        write the paired FASTA, predict the dimer structure, stage the
        best model + confidence JSON into canonical paths (renaming
        multi-char Boltz chains), then run the extractor.
        """
        seq_uid = record["seq_uid"]
        spec: BackboneSpec = pipeline.iter_seqs[backbone_id]

        # s3 — paired FASTA (designed sequence + target peptide).
        os.makedirs(pipeline.fasta_path, exist_ok=True)
        fasta_path = os.path.join(pipeline.fasta_path, f"{backbone_id}.fa")
        self._write_paired_fasta(fasta_path, backbone_id, record["sequence"], spec)

        # s4 — structure prediction (GPU). Output dir is per-backbone; the
        # nested boltz_results_<bid>/predictions/<bid>/ layout is the
        # predictor's, not ours.
        predict_out_dir = os.path.join(pipeline.dimer_models_path, backbone_id)
        await self.hooks.predict_structure(
            self.config, fasta_path, predict_out_dir
        )

        # s4_post_exec — stage best model + confidence JSON; rename chains.
        best_model_dst = os.path.join(pipeline.best_models_path, f"{backbone_id}.pdb")
        best_ptm_dst = os.path.join(pipeline.best_ptm_path, f"{backbone_id}.json")
        await self.hooks.stage_prediction(
            self.config,
            predict_out_dir,
            best_model_dst,
            best_ptm_dst,
            backbone_id,
        )

        # s5 — pLDDT/pTM/pAE extraction. The extractor reads from
        # ``<prediction_root>/best_models`` and ``<prediction_root>/best_ptm``
        # (parent of the two staging dirs).
        prediction_root = os.path.dirname(pipeline.best_models_path)
        csv_path = pipeline.stats_csv(cycle=pipeline.passes)
        results = await self.hooks.extract_metrics(
            self.config,
            pipeline.pipeline_id,
            pipeline.passes,
            prediction_root,
            csv_path,
        )
        # The extractor scans the staging dir and returns one row per
        # backbone present there; filter to the backbone we're currently
        # resolving so we don't attribute another structure's scores to
        # this one. The seq_uid the extractor synthesizes (= backbone_id)
        # gets replaced with the L1 candidate's real seq_uid.
        results = [r for r in results if r.backbone_id == backbone_id]
        for r in results:
            r.seq_uid = seq_uid
        return results

    @staticmethod
    def _write_paired_fasta(
        fasta_path: str,
        backbone_id: str,
        designed_sequence: str,
        spec: BackboneSpec,
    ) -> None:
        """Write the paired FASTA the structure predictor consumes.

        Layout matches IMPRESS's s3 stage:
            >designed_chain_name|<backbone_id>
            <designed_sequence>
            >target_chain_name|<backbone_id>
            <target_peptide>

        When ``spec.target_peptide`` is None (monomer mode) only the
        designed record is written.
        """
        with open(fasta_path, "w") as fd:
            fd.write(f">{spec.designed_chain_name}|{backbone_id}\n")
            fd.write(designed_sequence + "\n")
            if spec.target_peptide:
                fd.write(f">{spec.target_chain_name}|{backbone_id}\n")
                fd.write(spec.target_peptide + "\n")

    def _record_cycle(
        self,
        pipeline: ProteinBindingPipeline,
        cycle_metrics: List[PredictionResult],
    ) -> None:
        cycle_results = self._workflow_ddict.get("cycle_results", {}) or {}
        for pr in cycle_metrics:
            bid = pr.backbone_id
            bucket = cycle_results.get(bid, [])
            bucket.append(
                {
                    "pipeline_id": pipeline.pipeline_id,
                    "backbone_id": bid,
                    "cycle": pipeline.passes,
                    "seq_uid": pr.seq_uid,
                    "sequence": "",  # caller resolves from L1 buffer if needed
                    "produced_under_version": self._workflow_ddict.get(
                        "model_version", 0
                    ),
                    "prediction": pr.__dict__,
                }
            )
            cycle_results[bid] = bucket
            pipeline.current_scores[bid] = {
                "pLDDT": pr.pLDDT,
                "pTM": pr.pTM,
                "pAE": pr.pAE,
            }
        self._workflow_ddict["cycle_results"] = cycle_results

    # ------------------------------------------------------------------
    # training feedback loop
    # ------------------------------------------------------------------

    async def _schedule_training_round(self) -> None:
        """Curator callback: sample from corpus and dispatch the trainer hook."""
        try:
            sampled = self._sample_training_shard()
            ckpt_dir = os.path.join(
                self.config.mpnn_checkpoint_dir or self.config.base_path,
                f"mpnn_v{self._workflow_ddict.get('model_version', 0) + 1}",
            )
            await self.hooks.mpnn_train(self.config, sampled, ckpt_dir)
            self._workflow_ddict["mpnn_checkpoint_path"] = ckpt_dir
            self._workflow_ddict["model_version"] = (
                self._workflow_ddict.get("model_version", 0) + 1
            )
        finally:
            self._workflow_ddict["train_in_flight"] = False

    def _sample_training_shard(self) -> list:
        """Sample from the full corpus and return CorpusEntry-shaped dicts.

        Materialization (parquet write etc.) is the trainer hook's job;
        this keeps the orchestrator free of disk-format assumptions.
        """
        import random

        corpus = self._workflow_ddict.get("corpus", {}) or {}
        entries = list(corpus.values())
        if not entries:
            raise RuntimeError("training fired with empty corpus")

        k = min(self.config.train_shard_size, len(entries))
        if self.config.train_sampling == "weighted_by_score":
            weights = [
                (e["pTM"] * e["pLDDT"] / (1.0 + max(e["pAE"], 0.01)))
                for e in entries
            ]
            return random.choices(entries, weights=weights, k=k)
        return random.sample(entries, k=k)


# ---------------------------------------------------------------------------
# shared-state factory — Dragon DDict + Event, lazy-imported
# ---------------------------------------------------------------------------

def _new_shared_state():
    """Build (workflow_ddict, terminate_event).

    Dragon is the project's shared-state primitive; import lazily so this
    module is testable in environments where Dragon isn't installed.
    """
    try:
        from dragon.data.ddict import DDict  # type: ignore
        from dragon.native.event import Event  # type: ignore

        return DDict(), Event()
    except ImportError:
        return _DictStub(), _EventStub()


class _DictStub(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _EventStub:
    def __init__(self):
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True
