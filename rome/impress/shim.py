"""``RomeShim`` — attach ROME's corpus + continuous training to IMPRESS.

The shim is designed so the user's IMPRESS ``run_protein_binding.py``
changes by three lines:

1. construct a ``RomeShim`` next to the manager;
2. wrap the user's ``adaptive_decision`` with ``shim.wrap_adaptive_fn``;
3. enter ``async with shim.attached(manager):`` around ``manager.start(...)``.

Everything else — the ``ImpressManager``, ``PipelineSetup``,
``ProteinBindingPipeline``, the s1..s5 stage definitions — is untouched.
"""

import asyncio
import csv
import os
import random
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rome.protein.hooks import TaskHooks


@dataclass
class CorpusThresholds:
    """Score floors/ceilings a pass must clear to enter the training corpus.

    Defaults match the IMPRESS paper's "high confidence" cutoffs.
    """
    min_pLDDT: float = 80.0
    min_pTM: float = 0.80
    max_pAE: float = 5.0


CsvPathFn = Callable[[Any], str]
"""``(pipeline) -> str``. Returns the path of the per-pass score CSV.
Default mirrors IMPRESS's ``af_stats_<name>_pass_<passes>.csv``
relative-to-CWD convention."""


def _default_csv_path(pipeline) -> str:
    return f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv"


class RomeShim:
    """Additive ROME layer under an IMPRESS workflow.

    Parameters
    ----------
    corpus_thresholds : CorpusThresholds, optional
        Score gates for corpus membership. Default = paper's cutoffs.
    train_batch_threshold : int
        Number of new qualifying entries that must accumulate (since the
        last firing) before a training round is scheduled. Default 64.
    train_shard_size : int
        How many entries to sample per training round. Default 256.
    train_max_concurrent : int
        Cap on simultaneously running training tasks. Default 1.
    mpnn_checkpoint_dir : str, optional
        Where the trainer writes new checkpoints. Defaults to
        ``<base_path>/mpnn_ckpts``.
    base_path : str
        Used as the working dir for shim artifacts (parquet shards, etc.)
        and as the search root for the score CSVs when
        ``csv_path_for`` is unset.
    mpnn_train_config : dict, optional
        Forwarded to the foundry MPNN trainer. Read by the production
        ``mpnn_train_task`` from ``rome.protein.tasks``.
    task_hooks : TaskHooks, optional
        Pluggable science-tool implementations. Only ``mpnn_train`` is
        consulted by the shim; the rest are unused since IMPRESS owns
        the per-pass stages. Defaults to the production hooks.
    csv_path_for : Callable, optional
        Override how the shim locates the per-pass CSV for a pipeline.
        Default uses IMPRESS's relative-path convention.

    Attributes (read-only)
    ----------------------
    corpus_size : int
    model_version : int
        Bumps once per completed training round.
    current_checkpoint : str | None
        Path to the latest checkpoint written by the trainer.
    training_rounds : int
        Same as ``model_version``; aliased for readability.
    """

    def __init__(
        self,
        *,
        corpus_thresholds: Optional[CorpusThresholds] = None,
        train_batch_threshold: int = 64,
        train_shard_size: int = 256,
        train_max_concurrent: int = 1,
        mpnn_checkpoint_dir: Optional[str] = None,
        base_path: str = ".",
        mpnn_train_config: Optional[Dict[str, Any]] = None,
        task_hooks: Optional[TaskHooks] = None,
        csv_path_for: Optional[CsvPathFn] = None,
    ):
        self.thresholds = corpus_thresholds or CorpusThresholds()
        self.train_batch_threshold = train_batch_threshold
        self.train_shard_size = train_shard_size
        self.train_max_concurrent = train_max_concurrent
        self.base_path = base_path
        self.mpnn_checkpoint_dir = mpnn_checkpoint_dir or os.path.join(
            base_path, "mpnn_ckpts"
        )
        self.mpnn_train_config = mpnn_train_config or {}
        self.hooks = (task_hooks or TaskHooks()).resolved()
        self._csv_path_for = csv_path_for or _default_csv_path

        # Shared state
        self._corpus: Dict[str, Dict[str, Any]] = {}
        self._model_version = 0
        self._current_checkpoint: Optional[str] = None
        self._train_tasks: List[asyncio.Task] = []
        self._unconsumed_since_train = 0
        self._consumed_seq_uids: set = set()
        self._train_in_flight = 0  # int counter, not bool (supports concurrency cap)

    # ------------------------------------------------------------------
    # public read-only views
    # ------------------------------------------------------------------
    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def model_version(self) -> int:
        return self._model_version

    @property
    def current_checkpoint(self) -> Optional[str]:
        return self._current_checkpoint

    @property
    def training_rounds(self) -> int:
        return self._model_version

    # ------------------------------------------------------------------
    # adaptive_fn wrapping (IMPRESS's extension point)
    # ------------------------------------------------------------------
    def wrap_adaptive_fn(
        self,
        original_fn: Optional[Callable[[Any], Awaitable[Any]]],
    ) -> Callable[[Any], Awaitable[Any]]:
        """Wrap the user's IMPRESS ``adaptive_fn`` so ROME taps per-pass
        scores.

        IMPRESS calls ``adaptive_fn(pipeline)`` once per pass after the
        stage chain (s1..s5) completes. The wrapper:

        * runs the user's original logic first (so IMPRESS's view of
          ``current_scores`` / ``previous_scores`` is set normally),
        * then sweeps the just-written CSV into the ROME corpus,
        * schedules a training round if the threshold tripped.

        Returns ``None`` if the original returned ``None`` (IMPRESS's
        "continue normally" sentinel); otherwise returns whatever the
        original returned (typically a child-pipeline config).
        """
        async def wrapped(pipeline):
            original_result = None
            if original_fn is not None:
                original_result = await original_fn(pipeline)
            await self._harvest_pipeline(pipeline)
            return original_result

        return wrapped

    # ------------------------------------------------------------------
    # corpus harvest + training trigger
    # ------------------------------------------------------------------
    async def _harvest_pipeline(self, pipeline) -> None:
        csv_path = self._csv_path_for(pipeline)
        if not os.path.exists(csv_path):
            return
        new_entries = self._extract_passing_entries(csv_path, pipeline)
        for entry in new_entries:
            self._corpus[entry["pair_uid"]] = entry
            self._unconsumed_since_train += 1

        if (
            self._unconsumed_since_train >= self.train_batch_threshold
            and self._train_in_flight < self.train_max_concurrent
        ):
            self._train_in_flight += 1
            self._unconsumed_since_train = 0
            task = asyncio.create_task(self._fire_training())
            self._train_tasks.append(task)

    def _extract_passing_entries(self, csv_path: str, pipeline) -> List[Dict[str, Any]]:
        """Parse IMPRESS's score CSV (``ID, avg_plddt, ptm, avg_pae``).

        Filters by configured thresholds. Tracks consumed seq_uids so we
        don't double-count entries from re-read CSVs.
        """
        entries: List[Dict[str, Any]] = []
        try:
            with open(csv_path) as fd:
                reader = csv.DictReader(fd)
                for row in reader:
                    # Tolerate slight column-name drift in the upstream CSV.
                    ident = row.get("ID") or row.get("id")
                    if ident is None and row:
                        ident = next(iter(row.values()))
                    if ident is None or ident in self._consumed_seq_uids:
                        continue
                    try:
                        plddt = float(row["avg_plddt"])
                        ptm = float(row["ptm"])
                        pae = float(row["avg_pae"])
                    except (KeyError, ValueError):
                        continue

                    self._consumed_seq_uids.add(ident)

                    if plddt < self.thresholds.min_pLDDT:
                        continue
                    if ptm < self.thresholds.min_pTM:
                        continue
                    if pae > self.thresholds.max_pAE:
                        continue

                    backbone_id = ident.split(".")[0]
                    entries.append({
                        "pair_uid": str(uuid.uuid4()),
                        "backbone_id": backbone_id,
                        "pdb_path": self._pdb_path_for(pipeline, ident),
                        "sequence": "",  # IMPRESS's CSV doesn't carry the
                                         # sequence; the trainer adapter
                                         # resolves it from the PDB if needed.
                        "pLDDT": plddt,
                        "pTM": ptm,
                        "pAE": pae,
                        "produced_under_version": self._model_version,
                        "discovered_at_cycle": getattr(pipeline, "passes", 0),
                    })
        except FileNotFoundError:
            pass
        return entries

    def _pdb_path_for(self, pipeline, ident: str) -> str:
        """Best-effort guess at the PDB path for a CSV row.

        IMPRESS writes per-pipeline AF outputs under ``<name>_af/`` (main
        branch). The trainer hook will validate / discard if the file is
        missing — this string is only metadata.
        """
        af_dir = getattr(pipeline, "af_out_path", None)
        if af_dir is None:
            af_dir = f"{pipeline.name}_af"
        return os.path.join(af_dir, f"{ident}.pdb")

    async def _fire_training(self) -> None:
        """Sample a shard, dispatch the trainer hook, bump the version."""
        try:
            entries = list(self._corpus.values())
            if not entries:
                return
            k = min(self.train_shard_size, len(entries))
            sampled = random.sample(entries, k=k)
            new_version = self._model_version + 1
            ckpt_dir = os.path.join(
                self.mpnn_checkpoint_dir, f"mpnn_v{new_version}"
            )
            await self.hooks.mpnn_train(self, sampled, ckpt_dir)
            self._model_version = new_version
            self._current_checkpoint = ckpt_dir
        finally:
            self._train_in_flight = max(0, self._train_in_flight - 1)

    # ------------------------------------------------------------------
    # lifecycle — async context manager wrapping the IMPRESS manager
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def attached(self, manager=None):
        """Use the shim around ``manager.start(...)``.

        The shim itself is passive (no background loops); the context
        manager is for symmetry and so any in-flight training tasks
        get awaited before the run "officially" ends.

        ``manager`` is accepted but unused by the minimum-viable shim;
        a future streaming-MPNN extension may need to register a hook
        with the manager and would use this parameter.
        """
        try:
            yield self
        finally:
            if self._train_tasks:
                await asyncio.gather(*self._train_tasks, return_exceptions=True)
