"""Protein-binding workflow primitives.

Orchestration-only package: no torch/transformers/peft imports. Science tools
(foundry ProteinMPNN, AlphaFold2, the IMPRESS pLDDT/pTM/pAE extractor) are
invoked through ``rome.protein.tasks`` and are expected to be installed
separately — this package does not vendor or modify them.
"""

from rome.protein.config import ProteinBindingFlowConfig
from rome.protein.schema import (
    BackboneSpec,
    SequenceRecord,
    AF2Result,
    CycleResult,
    CorpusEntry,
)
from rome.protein.pipeline import ProteinBindingPipeline
from rome.protein.coordinator import AdaptiveCoordinator
from rome.protein.criteria import AdaptiveCriterion, default_criterion
from rome.protein.ranker import LogLikelihoodRanker
from rome.protein.corpus import CorpusCurator
from rome.protein.hooks import TaskHooks

__all__ = [
    "ProteinBindingFlowConfig",
    "BackboneSpec",
    "SequenceRecord",
    "AF2Result",
    "CycleResult",
    "CorpusEntry",
    "ProteinBindingPipeline",
    "AdaptiveCoordinator",
    "AdaptiveCriterion",
    "default_criterion",
    "LogLikelihoodRanker",
    "CorpusCurator",
    "TaskHooks",
]
