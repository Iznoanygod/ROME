"""Corpus curator — gates AF2 results into the MPNN training set.

Monotonic by design: once an entry passes the score thresholds it stays in
the corpus forever, regardless of which model version produced the sequence.
``produced_under_version`` is recorded for observability but never filters.

The curator owns the training trigger: when ``unconsumed_count`` crosses
``train_batch_threshold`` AND no training task is in flight, it fires
``schedule_train_fn`` once and resets the counter.
"""

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Optional  # noqa: F401

from rome.protein.schema import CorpusEntry, PredictionResult


class CorpusCurator:
    """Watches ``cycle_results`` and grows ``corpus``.

    Parameters
    ----------
    config : ProteinBindingFlowConfig
        Threshold + trigger knobs read from here.
    schedule_train_fn : Callable
        Awaitable invoked when a training round should start. The curator
        does not await it (fire-and-forget) so corpus growth keeps pace
        with the running training task.
    """

    def __init__(
        self,
        config: Any,
        schedule_train_fn: Callable[[], Awaitable[None]],
        poll_interval: float = 0.2,
        on_train_scheduled: Optional[Callable[[asyncio.Task], None]] = None,
    ):
        self.config = config
        self.schedule_train_fn = schedule_train_fn
        self.poll_interval = poll_interval
        # Optional callback so the orchestrator can track the spawned task
        # and await it on shutdown.
        self.on_train_scheduled = on_train_scheduled

    def _passes_thresholds(self, r: PredictionResult) -> bool:
        c = self.config
        if r.pLDDT < c.min_pLDDT_for_corpus:
            return False
        if r.pTM < c.min_pTM_for_corpus:
            return False
        if r.pAE > c.max_pAE_for_corpus:
            return False
        return True

    async def run(self, workflow_ddict: Any, terminate_event: Any) -> None:
        """Append qualifying entries; fire training when threshold trips."""
        self._consumed_seq_uids: set = set()
        self._unconsumed_count = 0
        while not terminate_event.is_set():
            self.sweep(workflow_ddict)
            await asyncio.sleep(self.poll_interval)

    def sweep(self, workflow_ddict: Any) -> None:
        """One pass: pull new cycle_results into corpus; fire training if due.

        Exposed so the orchestrator can force a final drain after pipelines
        finish — the background loop's sleep interval would otherwise race
        with shutdown on fast runs.
        """
        if not hasattr(self, "_consumed_seq_uids"):
            self._consumed_seq_uids = set()
            self._unconsumed_count = 0

        cycle_results = workflow_ddict.get("cycle_results", {}) or {}
        corpus = workflow_ddict.get("corpus", {}) or {}

        for backbone_id, summaries in cycle_results.items():
            for s in summaries:
                pr = s["prediction"]
                seq_uid = pr["seq_uid"]
                if seq_uid in self._consumed_seq_uids:
                    continue
                self._consumed_seq_uids.add(seq_uid)

                if not self._passes_thresholds(_as_prediction(pr)):
                    continue

                entry = CorpusEntry(
                    pair_uid=str(uuid.uuid4()),
                    backbone_id=backbone_id,
                    pdb_path=pr["pdb_path"],
                    sequence=s.get("sequence", ""),
                    pLDDT=pr["pLDDT"],
                    pTM=pr["pTM"],
                    pAE=pr["pAE"],
                    produced_under_version=s.get("produced_under_version", 0),
                    discovered_at_cycle=s["cycle"],
                )
                corpus[entry.pair_uid] = entry.__dict__
                self._unconsumed_count += 1

        workflow_ddict["corpus"] = corpus

        if (
            self.config.train_mpnn
            and self._unconsumed_count >= self.config.train_batch_threshold
            and not workflow_ddict.get("train_in_flight", False)
        ):
            workflow_ddict["train_in_flight"] = True
            self._unconsumed_count = 0
            task = asyncio.create_task(self.schedule_train_fn())
            if self.on_train_scheduled is not None:
                self.on_train_scheduled(task)


def _as_prediction(d: dict) -> PredictionResult:
    return PredictionResult(
        seq_uid=d["seq_uid"],
        backbone_id=d["backbone_id"],
        pdb_path=d["pdb_path"],
        pLDDT=d["pLDDT"],
        pTM=d["pTM"],
        pAE=d["pAE"],
        raw_csv_row=d.get("raw_csv_row"),
    )
