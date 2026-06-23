"""L1 log-likelihood ranker — streaming admission control before AF2.

Reads ``mpnn_outputs[backbone_id]`` (a list of :class:`SequenceRecord` accreted
by the MPNN generators) and maintains a per-backbone ranked candidate buffer
in ``ranked_candidates[backbone_id]``. Sequences are kept ordered by
descending log-likelihood; the AF2 scheduler pulls the head, the rest stay
available as fallbacks for the L2 escalation.

The ranker is intentionally stateless across pipeline restarts — it
recomputes from whatever lives in the workflow ddict.
"""

import asyncio
from typing import Any


class LogLikelihoodRanker:
    """Coroutine factory that keeps ``ranked_candidates`` in sync.

    Parameters
    ----------
    poll_interval : float
        Sleep between ddict scans. Cheap; the work itself is just a sort.
    """

    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval

    async def run(self, workflow_ddict: Any, terminate_event: Any) -> None:
        """Drain ``mpnn_outputs`` into sorted ``ranked_candidates``.

        Generators write into ``mpnn_outputs[backbone_id]``; the ranker
        moves them into ``ranked_candidates[backbone_id]`` (preserving any
        already-buffered items the L2 step hasn't yet consumed) and resorts
        by log-likelihood descending. ``mpnn_outputs`` is cleared as items
        are transferred so the generator back-pressure (which checks
        ``ranked_candidates`` length) is the only flow-control signal.
        """
        while not terminate_event.is_set():
            mpnn_outputs = workflow_ddict.get("mpnn_outputs", {}) or {}
            ranked = workflow_ddict.get("ranked_candidates", {}) or {}

            for backbone_id, records in list(mpnn_outputs.items()):
                if not records:
                    continue
                bucket = ranked.get(backbone_id, [])
                bucket.extend(records)
                bucket.sort(key=lambda r: r["log_likelihood"], reverse=True)
                ranked[backbone_id] = bucket
                # drain
                mpnn_outputs[backbone_id] = []

            workflow_ddict["mpnn_outputs"] = mpnn_outputs
            workflow_ddict["ranked_candidates"] = ranked
            await asyncio.sleep(self.poll_interval)
