"""L2 adaptive criterion — decides what to do with a backbone after AF2 scoring.

The criterion is the bridge between the cheap streaming L1 ranker and the
expensive sub-pipeline spawn from IMPRESS. It runs once per backbone per
cycle, after ``extract_metrics_task`` has produced a fresh score row.

The default policy mirrors IMPRESS's adaptive_decision: degraded -> migrate,
otherwise keep. The flow handles the "fallback to next-best from L1 buffer"
escalation outside of the criterion so the criterion stays a pure function
of current/previous scores.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional


class Decision(str, Enum):
    KEEP = "keep"             # accept current sequence, advance cycle
    FALLBACK = "fallback"     # try next-ranked L1 candidate
    MIGRATE = "migrate"       # hand backbone to a fresh sub-pipeline
    DROP = "drop"             # give up on this backbone


@dataclass
class CriterionInput:
    backbone_id: str
    current: dict          # metric_name -> float
    previous: Optional[dict]
    fallback_attempts: int
    sub_order: int
    config: "object"       # ProteinBindingFlowConfig — typed loosely to avoid cycle


# Pluggable signature.
AdaptiveCriterion = Callable[[CriterionInput], Awaitable[Decision]]


def _degraded(curr: dict, prev: dict) -> bool:
    """Paper's default: degradation = pAE up OR pTM down OR pLDDT down.

    Any single metric regressing trips the flag, matching IMPRESS's
    conservatism toward declining design trajectories.
    """
    if prev is None:
        return False
    if curr.get("pAE", 0.0) > prev.get("pAE", 0.0):
        return True
    if curr.get("pTM", 0.0) < prev.get("pTM", 0.0):
        return True
    if curr.get("pLDDT", 0.0) < prev.get("pLDDT", 0.0):
        return True
    return False


async def default_criterion(inp: CriterionInput) -> Decision:
    """IMPRESS-style default. Order of escalation:
    1. No previous score -> KEEP (first pass)
    2. Not degraded -> KEEP
    3. Degraded, fallback budget remains -> FALLBACK
    4. Degraded, fallback exhausted, sub-pipeline budget remains -> MIGRATE
    5. Otherwise -> DROP
    """
    if inp.previous is None:
        return Decision.KEEP
    if not _degraded(inp.current, inp.previous):
        return Decision.KEEP
    cfg = inp.config
    if inp.fallback_attempts < cfg.max_fallback_sequences:
        return Decision.FALLBACK
    if inp.sub_order < cfg.max_sub_pipelines:
        return Decision.MIGRATE
    return Decision.DROP
