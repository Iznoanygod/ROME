"""ProteinMPNN trainer task — the IMPRESS-R half of ROME-A's trainers.

IMPRESS runs backbone -> ProteinMPNN -> structure prediction -> pLDDT/pTM/pAE
-> keep/fallback/migrate/drop, and it is open loop: every campaign improves the
designs, never the model. IMPRESS-R closes that loop by feeding the campaign's
own high-confidence designs back into ProteinMPNN.

This module is the "one task" that adds ProteinMPNN to ROME-A. It targets
foundry's re-implementation (``RosettaCommons/foundry@production``,
``models/mpnn``), whose training entry point is
``models/mpnn/src/mpnn/train.py``.

The shape of the training data is the thing to understand before reading on.
ProteinMPNN is *inverse folding* — structure in, sequence out — so the label is
the sequence embedded in the structure file, and a foundry training set is a
**dataframe of metadata rows pointing at structure files**, not a table of
sequences. :func:`build_training_dataframe` is where a ROME-A corpus becomes
that. See ``docs/proteinmpnn_training.md`` for the full derivation, the column
requirements, and the open questions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from rome.train.base import TrainTask

#: Corpus fields a design must carry to be trainable. ``path`` is the structure
#: file that *is* the training example; ``sequence`` is needed only to size it.
REQUIRED_FIELDS = ("path", "sequence")

#: Proteins at or below this length are counted as peptides by the AF3-style
#: example weighting (``atomworks.ml.preprocessing.constants.PEPTIDE_MAX_RESIDUES``).
PEPTIDE_MAX_RESIDUES = 20

#: Filters that make sense for a design campaign. The stock ``MPNN_FILTERS`` in
#: foundry's train.py additionally reference ``resolution``, ``method`` and
#: ``deposition_date``, which a campaign has no analogue for — and since
#: PandasDataset filters are pandas query strings, a missing column is a hard
#: error rather than a no-op.
DEFAULT_FILTERS = (
    "n_non_atomized_tokens >= 30",
    "n_prot == 1",
    "cluster.notnull() and cluster != 'nan'",
)

#: AF3-style example weighting alphas. Protein-only, matching foundry's
#: ProteinMPNN settings.
DEFAULT_ALPHAS = {"a_prot": 1.0, "a_nuc": 0.0, "a_ligand": 0.0, "a_loi": 0.0}


def generate_example_id(
    dataset_names: List[str], pdb_id: str, assembly_id: str, query_pn_unit_iids: List[str]
) -> str:
    """Reproduce ``atomworks.ml.example_id.generate_example_id``.

    Inlined so a corpus dataframe can be built (and tested) without importing
    atomworks, which pulls in the whole structural-ML stack. The real function
    is used instead when atomworks is importable, so the two cannot drift.
    """
    try:
        from atomworks.ml.example_id import generate_example_id as _real

        return _real(dataset_names, pdb_id, assembly_id, query_pn_unit_iids)
    except ImportError:
        return f"{{{dataset_names}}}{{{pdb_id}}}{{{assembly_id}}}{{{query_pn_unit_iids}}}"


@dataclass
class ProteinMPNNConfig:
    """Configuration for fine-tuning ProteinMPNN on campaign results.

    Defaults describe a *short mid-campaign round*, not a pretraining run.
    foundry's reference script trains from scratch (``max_epochs=500``,
    ``n_examples_per_epoch=20000``); a ROME-A round wants to touch the corpus a
    handful of times and publish.

    Parameters
    ----------
    model_type : str
        ``'protein_mpnn'`` or ``'ligand_mpnn'``.
    initial_checkpoint : Optional[str]
        Checkpoint the first round starts from, in foundry format (a dict with
        a ``"model"`` key). Public ProteinMPNN weights are in the *legacy*
        format and must be converted once — see
        :func:`convert_legacy_checkpoint`. Later rounds start from whatever the
        previous round published, which the training manager passes in as
        ``model_path``.
    dataset_name : str
        Name embedded in each example id, identifying the campaign.
    filters : Sequence[str]
        pandas query strings applied to the corpus dataframe.
    alphas : dict
        AF3-style example-weighting alphas.
    cluster_by : str
        Corpus field used as the clustering key. Weight is
        ``beta / cluster_size``, so this is what stops one prolific backbone
        from dominating a round; ``backbone_id`` gives each backbone equal total
        weight regardless of how many of its designs passed the filter.
    max_tokens_with_padding : int
        Token budget per batch. foundry uses 10000 for ProteinMPNN.
    max_epochs, n_examples_per_epoch : int
        Round length. ``n_examples_per_epoch=None`` means "the corpus size".
    train_structure_noise : float
        Gaussian coordinate noise for augmentation. foundry uses 0.2 for
        ProteinMPNN.
    devices_per_node, num_nodes, accelerator, precision : Any
        Passed to the foundry trainer.
    num_workers : int
        DataLoader workers.
    cache_dir : Optional[str]
        atomworks structure-parse cache. Strongly worth setting: the same
        design is re-parsed every round it survives in the corpus.
    reset_optimizer : bool
        Reset optimizer/scheduler state when resuming. Leave ``False`` between
        rounds — the Noam schedule is step-based, and resetting it re-runs
        warm-up every round. Forced ``True`` for a converted legacy checkpoint,
        which carries no optimizer state.
    learning_rate_factor, warmup_steps, d_model : float / int / int
        Noam schedule parameters.
    shard_dir : Optional[str]
        Where the round's corpus dataframe is written. Defaults to a ``shards``
        subdirectory of the round's output directory.
    trainer_kwargs, train_func : ...
        Escape hatches — see the class docstring of :class:`ProteinMPNNTrainer`.
    """

    model_type: str = "protein_mpnn"
    initial_checkpoint: Optional[str] = None
    dataset_name: str = "impress_r"
    filters: Sequence[str] = DEFAULT_FILTERS
    alphas: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ALPHAS))
    beta: float = 1.0
    cluster_by: str = "backbone_id"
    max_tokens_with_padding: int = 10000
    max_epochs: int = 3
    n_examples_per_epoch: Optional[int] = None
    train_structure_noise: float = 0.2
    accelerator: str = "gpu"
    devices_per_node: Any = 1
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    num_workers: int = 4
    cache_dir: Optional[str] = None
    reset_optimizer: bool = False
    learning_rate_factor: float = 2.0
    warmup_steps: int = 4000
    d_model: int = 128
    clip_grad_max_norm: Optional[float] = None
    shard_dir: Optional[str] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    train_func: Optional[Callable[..., str]] = None

    def validate(self) -> None:
        if self.model_type not in ("protein_mpnn", "ligand_mpnn"):
            raise ValueError(
                f"unknown model_type {self.model_type!r}; "
                "expected 'protein_mpnn' or 'ligand_mpnn'"
            )


# ---------------------------------------------------------------------------
# Corpus -> foundry training dataframe
# ---------------------------------------------------------------------------

def build_training_dataframe(
    records: List[Dict[str, Any]],
    *,
    dataset_name: str = "impress_r",
    cluster_by: str = "backbone_id",
) -> Any:
    """Turn a ROME-A corpus into the metadata dataframe foundry trains on.

    One row per design. The row does not contain the sequence as a label — it
    points at a structure file, and the sequence atomworks reads out of that
    file is the label. For IMPRESS-R the natural file is the structure
    prediction of the designed sequence: its coordinates and its sequence are
    already the pair we want to train on, with no threading step.

    Columns follow foundry's ``GenericDFParser`` plus everything the AF3-style
    example weighting reads. Note ``n_peptide``: the weighting asserts on
    ``n_prot``/``n_nuc``/``n_ligand``/``cluster_size`` and then reads
    ``n_peptide`` unguarded, so omitting it raises a bare ``KeyError``.

    Parameters
    ----------
    records : list of dict
        Corpus records. Each needs ``path`` and ``sequence``; ``backbone_id``,
        ``pLDDT``, ``pTM``, ``pAE`` and ``uid`` are carried through when
        present, and reach the transform pipeline as ``extra_info``.
    dataset_name : str
        Embedded in each example id, identifying the campaign.
    cluster_by : str
        Record field used as the cluster key, falling back to the design's own
        id when absent (i.e. every design its own cluster, uniform weighting).

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    rows = []
    for record in records:
        design_id = str(record.get("uid") or record.get("design_id") or record["path"])
        sequence = record.get("sequence") or ""
        n_tokens = int(record.get("n_non_atomized_tokens") or len(sequence))
        cluster = record.get(cluster_by) or design_id

        rows.append(
            {
                # --- GenericDFParser ---
                "example_id": generate_example_id(
                    [dataset_name], design_id, "1", ["A_1"]
                ),
                "path": os.path.abspath(record["path"]),
                "assembly_id": "1",
                # --- filters + token-budget batching ---
                "n_non_atomized_tokens": n_tokens,
                # --- AF3-style example weighting ---
                "cluster": str(cluster),
                "n_prot": 1,
                "n_peptide": 1 if n_tokens <= PEPTIDE_MAX_RESIDUES else 0,
                "n_nuc": 0,
                "n_ligand": 0,
                "q_pn_unit_is_loi": 0,
                # --- carried through as extra_info; also the audit trail ---
                "design_id": design_id,
                "sequence": sequence,
                "backbone_id": record.get("backbone_id"),
                "pLDDT": record.get("pLDDT"),
                "pTM": record.get("pTM"),
                "pAE": record.get("pAE"),
                "produced_under_version": record.get("model_version"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checkpoint bootstrap
# ---------------------------------------------------------------------------

def convert_legacy_checkpoint(weights_path: str, output_path: str,
                              model_type: str = "protein_mpnn") -> str:
    """Convert public ProteinMPNN/LigandMPNN weights into foundry format.

    The released weights (e.g. ``proteinmpnn_v_48_020.pt``) predate foundry's
    re-implementation: parameter names differ, an unused atom-type embedding
    has to be dropped, the pairwise backbone-distance embedding is permuted,
    and the amino-acid vocabulary is reordered (legacy is alphabetical by
    one-letter code, foundry by three-letter). ``mpnn.utils.weights.load_legacy_weights``
    does all of that, but into a live model rather than into a file — so
    bootstrapping round one is a one-time conversion.

    The result is a ``{"model": state_dict}`` checkpoint, which is what both
    the trainer's resume path and ``MPNNInferenceEngine`` expect. Load it with
    ``reset_optimizer=True``: there is no optimizer state to restore.
    """
    import torch
    from mpnn.model.mpnn import LigandMPNN, ProteinMPNN  # type: ignore
    from mpnn.utils.weights import load_legacy_weights  # type: ignore

    model = ProteinMPNN() if model_type == "protein_mpnn" else LigandMPNN()
    load_legacy_weights(model, weights_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, output_path)
    return output_path


def latest_checkpoint_file(output_dir: str) -> Optional[str]:
    """Newest ``epoch-NNNN.ckpt`` written by a round, or ``None``.

    Deliberately *not* ``FabricTrainer.get_latest_checkpoint``, which returns
    ``sorted(dir.iterdir())[-1]`` — every file in the directory, not just
    checkpoints — so a stray log or temp file silently becomes "the latest
    checkpoint". A file is also what has to be published: the inference engine
    validates ``ckpt_path.is_file()`` and will reject a directory.
    """
    ckpt_dir = os.path.join(output_dir, "ckpt")
    if not os.path.isdir(ckpt_dir):
        return None
    checkpoints = sorted(
        f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")
    )
    return os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else None


# ---------------------------------------------------------------------------
# The training task
# ---------------------------------------------------------------------------

class ProteinMPNNTrainer(TrainTask):
    """ProteinMPNN trainer task in the ROME framework.

    Fine-tunes ProteinMPNN on the campaign's own high-confidence designs and
    returns the checkpoint file, which the training manager publishes and the
    inference streams reload.

    Set ``config.train_func`` to bypass the foundry path entirely and call your
    own trainer with ``(dataframe_path, output_dir, config) -> checkpoint_path``
    — useful for sites running a ProteinMPNN fork, and for testing.

    Parameters
    ----------
    config : ProteinMPNNConfig, optional
    gpus, nodes : int
        Resources one round needs, forwarded to asyncflow. Defaults follow
        ``config.devices_per_node`` / ``config.num_nodes``.
    """

    def __init__(
        self,
        config: Optional[ProteinMPNNConfig] = None,
        *,
        gpus: Optional[int] = None,
        nodes: Optional[int] = None,
        name: Optional[str] = None,
    ):
        config = config or ProteinMPNNConfig()
        config.validate()
        devices = config.devices_per_node if isinstance(config.devices_per_node, int) else 1
        super().__init__(
            gpus=gpus if gpus is not None else devices,
            nodes=nodes if nodes is not None else config.num_nodes,
            name=name or "proteinmpnn",
        )
        self.config = config

    #: Records stay plain dicts — foundry reads a dataframe of file paths, not a
    #: HuggingFace dataset.
    wants_hf_dataset = False

    def validate(self, dataset: Any) -> None:
        super().validate(dataset)
        missing = [f for f in REQUIRED_FIELDS if f not in (dataset[0] or {})]
        if missing:
            raise ValueError(
                f"ProteinMPNN training needs {', '.join(REQUIRED_FIELDS)} on every "
                f"record; the corpus is missing {', '.join(missing)}. 'path' must "
                "point at the structure file that IS the training example (for "
                "IMPRESS-R, the structure prediction of the designed sequence) — "
                "see docs/proteinmpnn_training.md."
            )

    # -- corpus materialization --------------------------------------------

    def write_shard(self, records: List[Dict[str, Any]], output_dir: str) -> str:
        """Write the round's training dataframe to parquet; returns its path.

        Public and side-effect-light so a workflow can inspect exactly what a
        round trained on — the shard is the audit trail for "the campaign
        improved the model that generates the designs".
        """
        shard_dir = self.config.shard_dir or os.path.join(output_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        frame = build_training_dataframe(
            records,
            dataset_name=self.config.dataset_name,
            cluster_by=self.config.cluster_by,
        )
        path = os.path.join(shard_dir, "train.parquet")
        frame.to_parquet(path)
        return path

    # -- the round ----------------------------------------------------------

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Fine-tune ProteinMPNN on the campaign's designs.

        ``kwargs`` carries ``model_version`` from the training manager, and
        ``model_path`` when a previous round has published one.

        Returns the ``.ckpt`` file, not the directory: ``MPNNInferenceEngine``
        validates that its ``checkpoint_path`` is a file.
        """
        records = list(dataset)
        shard_path = self.write_shard(records, output_dir)

        if self.config.train_func is not None:
            return self.config.train_func(shard_path, output_dir, self.config) or output_dir

        return self._train_with_foundry(records, shard_path, output_dir, **kwargs)

    def _train_with_foundry(self, records, shard_path, output_dir, **kwargs) -> str:
        """One foundry training round.

        Mirrors ``models/mpnn/src/mpnn/train.py``, with campaign-appropriate
        filters and a round-length rather than pretraining-length schedule.
        Every heavyweight import happens here, on the node with the GPUs.
        """
        import pandas as pd
        from atomworks.io.parser import STANDARD_PARSER_ARGS
        from atomworks.ml.datasets.pandas_dataset import (
            PandasDataset,
            StructuralDatasetWrapper,
        )
        from atomworks.ml.datasets.parsers.default_metadata_row_parsers import (
            GenericDFParser,
        )
        from atomworks.ml.samplers import calculate_weights_for_pdb_dataset_df
        from mpnn.collate.feature_collator import TokenBudgetAwareFeatureCollator  # type: ignore
        from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline  # type: ignore
        from mpnn.samplers.samplers import PaddedTokenBudgetBatchSampler  # type: ignore
        from mpnn.trainers.mpnn import MPNNTrainer  # type: ignore
        from omegaconf import DictConfig
        from torch.utils.data import DataLoader, WeightedRandomSampler

        from foundry.utils.weights import CheckpointConfig  # type: ignore

        cfg = self.config
        frame = pd.read_parquet(shard_path)

        parser_args = dict(STANDARD_PARSER_ARGS)
        if cfg.cache_dir:
            # The same design is re-parsed every round it survives in the
            # corpus, so caching is worth real time across a campaign.
            parser_args.update(
                load_from_cache=True, save_to_cache=True, cache_dir=cfg.cache_dir
            )

        structural_dataset = StructuralDatasetWrapper(
            dataset=PandasDataset(
                data=frame,
                id_column="example_id",
                name=cfg.dataset_name,
                filters=list(cfg.filters),
            ),
            dataset_parser=GenericDFParser(
                example_id_colname="example_id",
                path_colname="path",
                assembly_id_colname="assembly_id",
            ),
            transform=build_mpnn_transform_pipeline(
                model_type=cfg.model_type,
                is_inference=False,
                minimal_return=True,
                train_structure_noise_default=cfg.train_structure_noise,
            ),
            cif_parser_args=parser_args,
        )

        weights = calculate_weights_for_pdb_dataset_df(
            dataset_df=structural_dataset.data, beta=cfg.beta, alphas=cfg.alphas
        )
        sampler = WeightedRandomSampler(weights, len(structural_dataset))
        batch_sampler = PaddedTokenBudgetBatchSampler(
            sampler=sampler,
            get_num_tokens=lambda idx: structural_dataset.data.iloc[
                idx[0] if isinstance(idx, (list, tuple)) else idx
            ]["n_non_atomized_tokens"],
            max_tokens_with_padding=cfg.max_tokens_with_padding,
            shuffle_batches=True,
        )
        train_loader = DataLoader(
            structural_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            collate_fn=TokenBudgetAwareFeatureCollator(
                max_tokens_with_padding=cfg.max_tokens_with_padding
            ),
        )

        trainer = MPNNTrainer(
            model_type=cfg.model_type,
            accelerator=cfg.accelerator,
            devices_per_node=cfg.devices_per_node,
            num_nodes=cfg.num_nodes,
            precision=cfg.precision,
            max_epochs=cfg.max_epochs,
            n_examples_per_epoch=cfg.n_examples_per_epoch or max(1, len(structural_dataset)),
            output_dir=output_dir,
            clip_grad_max_norm=cfg.clip_grad_max_norm,
            # Checkpoint every epoch, so a short round always leaves something
            # to publish.
            checkpoint_every_n_epochs=1,
            **cfg.trainer_kwargs,
        )

        trainer.initialize_or_update_trainer_state(
            {"train_cfg": DictConfig({"model": {
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    # Absolute LR is set by the Noam schedule, not here.
                    "lr": 1.0,
                    "betas": [0.9, 0.98],
                    "eps": 1e-9,
                    "weight_decay": 0.0,
                },
                "lr_scheduler": {
                    "_target_": "rome.train.mpnn.create_noam_scheduler",
                    "d_model": cfg.d_model,
                    "warmup_steps": cfg.warmup_steps,
                    "factor": cfg.learning_rate_factor,
                },
            }})}
        )
        trainer.fabric.launch()
        trainer.construct_model()
        trainer.construct_optimizer()
        trainer.construct_scheduler()

        ckpt_config = self._resume_config(CheckpointConfig, **kwargs)
        trainer.fit(train_loader=train_loader, val_loaders=None, ckpt_config=ckpt_config)

        checkpoint = latest_checkpoint_file(output_dir)
        if checkpoint is None:
            raise RuntimeError(
                f"{self.name}: training finished but no checkpoint was written under "
                f"{output_dir}/ckpt. Check that max_epochs >= 1 and that the corpus "
                "survived the configured filters."
            )
        return checkpoint

    def _resume_config(self, checkpoint_config_cls, **kwargs):
        """Where this round starts from.

        Rounds after the first continue from the previously published
        checkpoint *with optimizer and scheduler state intact* — the Noam
        schedule is step-based, so resetting it would re-run warm-up every
        round. A converted legacy checkpoint carries no optimizer state, so
        that one case forces a reset.
        """
        previous = kwargs.get("model_path")
        if previous:
            return checkpoint_config_cls(
                path=previous,
                weight_loading_config=None,
                reset_optimizer=self.config.reset_optimizer,
            )
        if self.config.initial_checkpoint:
            return checkpoint_config_cls(
                path=self.config.initial_checkpoint,
                weight_loading_config=None,
                reset_optimizer=True,
            )
        return None


def create_noam_scheduler(
    optimizer: Any, d_model: int, warmup_steps: int = 4000, factor: float = 2.0
) -> Any:
    """NoamOpt-style LR schedule, as foundry's reference script builds it.

    ``lr = factor * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)``.
    Referenced by ``_target_`` from the trainer's ``train_cfg``, so it has to be
    importable by path from the training task.
    """
    import torch

    def noam_lambda(step: int) -> float:
        if step == 0:
            return 0.0
        return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)


def impress_corpus_filter(
    min_pLDDT: float = 80.0,
    min_pTM: float = 0.8,
    max_pAE: float = 5.0,
) -> Callable[[Dict[str, Any]], bool]:
    """Build the IMPRESS admission predicate for :class:`~rome.data.DataConfig`.

    IMPRESS-R only trains on designs the campaign is confident in, and those are
    exactly the thresholds IMPRESS already uses to decide a design is worth
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


__all__ = [
    "ProteinMPNNConfig",
    "ProteinMPNNTrainer",
    "build_training_dataframe",
    "convert_legacy_checkpoint",
    "latest_checkpoint_file",
    "create_noam_scheduler",
    "impress_corpus_filter",
    "generate_example_id",
    "PEPTIDE_MAX_RESIDUES",
    "DEFAULT_FILTERS",
    "DEFAULT_ALPHAS",
]
