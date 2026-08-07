"""ProteinMPNN trainer task — the IMPRESS-R half of ROME-A's trainers.

IMPRESS runs backbone -> ProteinMPNN -> structure prediction -> pLDDT/pTM/pAE
-> keep/fallback/migrate/drop, and it is open loop: every campaign improves the
designs, never the model. IMPRESS-R closes that loop by feeding the campaign's
own high-confidence designs back into ProteinMPNN.

This module is the "one task" that adds ProteinMPNN to ROME-A. It takes corpus
records the data manager collected — ``(sequence, pdb_path, pLDDT, pTM, pAE)``
tuples the campaign produced — materializes them as a training shard, and runs
the MPNN trainer over them. Everything else (when to train, which designs
qualify, who picks up the checkpoint) is generic ROME-A machinery.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from rome.train.base import TrainTask

#: Corpus fields a design has to carry to be trainable.
REQUIRED_FIELDS = ("sequence", "pdb_path")


@dataclass
class ProteinMPNNConfig:
    """Configuration for fine-tuning ProteinMPNN on campaign results.

    Parameters
    ----------
    weights_dir : Optional[str]
        Checkpoint the first round starts from. Later rounds start from the
        checkpoint the previous round published, which the training manager
        passes in as ``model_path``.
    backend : str
        ``'foundry'`` uses ``mpnn.trainers.mpnn.MPNNTrainer``; ``'custom'``
        calls :attr:`train_func` instead, which is the escape hatch for sites
        running their own ProteinMPNN fork.
    train_func : Optional[Callable]
        ``(shard_path, output_dir, config) -> checkpoint_path``. Required when
        ``backend='custom'``.
    shard_format : str
        ``'parquet'`` (foundry's expected input) or ``'jsonl'``. JSONL needs no
        pyarrow, which makes it the practical choice for dry runs.
    shard_dir : Optional[str]
        Where training shards are written. Defaults to a ``shards``
        subdirectory of the round's output directory.
    trainer_args : dict
        Passed straight to the MPNN trainer (epochs, learning rate, ...).
    fields : Sequence[str]
        Corpus fields carried into the shard. Extra fields are dropped so a
        wide corpus does not bloat the training input.
    """

    weights_dir: Optional[str] = None
    backend: str = "foundry"
    train_func: Optional[Callable[..., str]] = None
    shard_format: str = "parquet"
    shard_dir: Optional[str] = None
    trainer_args: Dict[str, Any] = field(default_factory=dict)
    fields: Sequence[str] = (
        "sequence",
        "pdb_path",
        "backbone_id",
        "pLDDT",
        "pTM",
        "pAE",
        "score",
        "model_version",
    )

    def validate(self) -> None:
        if self.backend not in ("foundry", "custom"):
            raise ValueError(
                f"unknown ProteinMPNN backend {self.backend!r}; "
                "expected 'foundry' or 'custom'"
            )
        if self.backend == "custom" and self.train_func is None:
            raise ValueError("ProteinMPNNConfig.train_func is required for backend='custom'")
        if self.shard_format not in ("parquet", "jsonl"):
            raise ValueError(
                f"unknown shard_format {self.shard_format!r}; expected 'parquet' or 'jsonl'"
            )


class ProteinMPNNTrainer(TrainTask):
    """ProteinMPNN trainer task in the ROME framework.

    Parameters
    ----------
    config : ProteinMPNNConfig, optional
        Defaults train through foundry from whatever checkpoint the training
        manager has published.
    gpus, nodes : int
        Resources one round needs, forwarded to asyncflow.
    """

    def __init__(
        self,
        config: Optional[ProteinMPNNConfig] = None,
        *,
        gpus: int = 1,
        nodes: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(gpus=gpus, nodes=nodes, name=name or "proteinmpnn")
        self.config = config or ProteinMPNNConfig()
        self.config.validate()

    #: Records stay plain dicts — the MPNN trainers read a shard file, not a
    #: HuggingFace dataset.
    wants_hf_dataset = False

    def validate(self, dataset: Any) -> None:
        super().validate(dataset)
        missing = [f for f in REQUIRED_FIELDS if f not in (dataset[0] or {})]
        if missing:
            raise ValueError(
                f"ProteinMPNN training needs {', '.join(REQUIRED_FIELDS)} on every "
                f"record; the corpus is missing {', '.join(missing)}. Add them via "
                "add_training_data(sequence=..., pdb_path=..., score=plddt)."
            )

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Fine-tune ProteinMPNN on the campaign's own best designs.

        ``kwargs`` carries ``model_version`` from the training manager; the
        starting checkpoint is the previously published one when there is one,
        falling back to ``config.weights_dir``.
        """
        shard_path = self.write_shard(dataset, output_dir)
        start_from = kwargs.get("model_path") or self.config.weights_dir

        if self.config.backend == "custom":
            result = self.config.train_func(shard_path, output_dir, self.config)
            return result or output_dir

        from mpnn.trainers.mpnn import MPNNTrainer  # type: ignore

        trainer = MPNNTrainer(
            train_data=shard_path,
            output_dir=output_dir,
            checkpoint=start_from,
            **self.config.trainer_args,
        )
        trainer.fit()
        return output_dir

    # -- shard materialization ---------------------------------------------

    def write_shard(self, records: List[Dict[str, Any]], output_dir: str) -> str:
        """Write the corpus slice the MPNN trainer will read.

        Kept public and side-effect-light so a workflow can inspect exactly
        what a round trained on — the shard is the audit trail for "the
        campaign improved the model that generates the designs".
        """
        shard_dir = self.config.shard_dir or os.path.join(output_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        rows = [self._project(r) for r in records]

        if self.config.shard_format == "jsonl":
            path = os.path.join(shard_dir, f"shard_{uuid.uuid4().hex[:8]}.jsonl")
            with open(path, "w") as fd:
                for row in rows:
                    fd.write(json.dumps(row) + "\n")
            return path

        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        path = os.path.join(shard_dir, f"shard_{uuid.uuid4().hex[:8]}.parquet")
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def _project(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {f: record.get(f) for f in self.config.fields if f in record}


def impress_corpus_filter(
    min_pLDDT: float = 80.0,
    min_pTM: float = 0.8,
    max_pAE: float = 5.0,
) -> Callable[[Dict[str, Any]], bool]:
    """Build the IMPRESS admission predicate for :class:`~rome.data.DataConfig`.

    IMPRESS-R only trains on designs the campaign is confident in, and those
    are exactly the thresholds IMPRESS already uses to decide a design is worth
    keeping::

        DataConfig(min_samples=64, filter_func=impress_corpus_filter())
    """

    def _passes(record: Dict[str, Any]) -> bool:
        plddt = record.get("pLDDT", record.get("score"))
        if plddt is None or plddt < min_pLDDT:
            return False
        if record.get("pTM", min_pTM) < min_pTM:
            return False
        if record.get("pAE", max_pAE) > max_pAE:
            return False
        return True

    return _passes


__all__ = ["ProteinMPNNConfig", "ProteinMPNNTrainer", "impress_corpus_filter"]
