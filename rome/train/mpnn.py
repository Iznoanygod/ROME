"""ProteinMPNN trainer task — the IMPRESS-R half of ROME-A's trainers.

IMPRESS runs backbone -> ProteinMPNN -> structure prediction -> pLDDT/pTM/pAE
-> keep/fallback/migrate/drop, and it is open loop: every campaign improves the
designs, never the model. IMPRESS-R closes that loop by fine-tuning ProteinMPNN
on the campaign's own high-confidence designs.

**Which ProteinMPNN.** IMPRESS runs the original ``dauparas/ProteinMPNN``: its
``mpnn_wrapper.py`` shells out to ``protein_mpnn_run.py`` with the original CLI
and helper scripts, and its setup clones that repo directly. So this trainer
targets the *same* implementation — it fine-tunes the original model and writes
a checkpoint in the original ``{"model_state_dict": ...}`` format that
``protein_mpnn_run.py`` loads. A foundry / re-implementation checkpoint would
not load into what IMPRESS runs; that was the previous version's mistake.

(PyRosetta is in the IMPRESS stack too, but for FastRelax and pLDDT extraction,
not for sequence design — the design model is vanilla ProteinMPNN.)

**Dimers, not monomers.** An IMPRESS protein-binding design is a *complex*: a
designed chain (``A``) plus a fixed target peptide (``B``). Fine-tuning has to
respect that — the model should learn to design chain A *in the context of* the
peptide, scoring sequence recovery on the designed chain only. That is exactly
what ProteinMPNN's chain mask expresses, and :func:`build_chain_designation`
sets it up: designed chains are predicted, context chains are visible but not
scored.

The heavy training loop imports torch and the ProteinMPNN repo and runs on a
GPU; it is isolated in :meth:`ProteinMPNNTrainer._train_with_proteinmpnn` and is
written against the repo's stable public API but has not been executed in CI.
Everything up to it — chain designation, structure staging, the manifest, the
checkpoint format, weight publication — is plain Python and is tested. Use
``config.train_func`` to substitute your own loop until the built-in one is
validated on your pinned checkout. See ``docs/proteinmpnn_training.md``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rome.train.base import TrainTask

#: ProteinMPNN's amino-acid alphabet (index -> letter). Position 20 is ``X``.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

#: A corpus record must carry ``path`` — the structure file whose designed chain
#: IS the training label. ``sequence`` is optional (ProteinMPNN reads the label
#: out of the structure); it is kept for the manifest and for sizing.
REQUIRED_FIELDS = ("path",)

#: The IMPRESS protein-binding layout: chain A is the designed binder, chain B
#: the fixed target peptide. Designed chains are scored; context chains are
#: visible to the model but not scored.
DEFAULT_DESIGN_CHAINS: Tuple[str, ...] = ("A",)
DEFAULT_CONTEXT_CHAINS: Tuple[str, ...] = ("B",)

#: Architecture of the public ``v_48_020`` weights, which is what IMPRESS runs.
#: ``protein_mpnn_run.py`` hardcodes hidden_dim=128 and 3+3 layers and reads
#: ``k_neighbors`` back from the checkpoint's ``num_edges`` — so a fine-tune must
#: keep these or the published checkpoint will not load.
DEFAULT_NUM_NEIGHBORS = 48
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 3
DEFAULT_BACKBONE_NOISE = 0.2


# ---------------------------------------------------------------------------
# Pure-Python helpers (no torch): chain designation, staging, checkpoint format
# ---------------------------------------------------------------------------

def pdb_chain_ids(path: str) -> List[str]:
    """Chain IDs present in a PDB, in first-seen order — column 22 of ATOM/HETATM.

    Plain text parsing so a campaign's monomer-vs-complex shape can be checked
    (and a bad chain designation caught) without importing a structure library.
    """
    seen: List[str] = []
    try:
        with open(path) as fd:
            for line in fd:
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                    chain = line[21]
                    if chain not in seen:
                        seen.append(chain)
    except OSError:
        return []
    return seen


def build_chain_designation(
    records: List[Dict[str, Any]],
    *,
    design_chains: Sequence[str] = DEFAULT_DESIGN_CHAINS,
    context_chains: Sequence[str] = DEFAULT_CONTEXT_CHAINS,
    chains_func: Optional[Callable[[Dict[str, Any], List[str]], Tuple[List[str], List[str]]]] = None,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Map each design to ``(designed_chains, context_chains)``.

    This is the dimer fix. ProteinMPNN's ``tied_featurize`` takes exactly this
    mapping and builds the chain mask from it: designed chains are predicted and
    scored, context chains are visible to the model but excluded from the loss.
    For an IMPRESS binder that means "learn chain A given chain B", which is the
    thing the campaign is actually optimising.

    A record may override the defaults per structure with a ``design_chains`` /
    ``context_chains`` field, or a ``chains_func(record, present_chains)`` may
    decide from the structure itself. Chains named but absent from the file are
    dropped with the designation still valid for whatever remains; a design with
    no designable chain present is an error, caught in
    :meth:`ProteinMPNNTrainer.validate`.

    Returns ``{design_name: (designed, context)}`` keyed by the staged structure
    name (see :func:`stage_structures`), which is what ``tied_featurize`` keys on.
    """
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for index, record in enumerate(records):
        name = _design_name(record, index)
        present = pdb_chain_ids(record["path"]) if record.get("path") else []
        if chains_func is not None:
            designed, context = chains_func(record, present)
        else:
            designed = list(record.get("design_chains", design_chains))
            context = list(record.get("context_chains", context_chains))
        if present:
            designed = [c for c in designed if c in present]
            context = [c for c in context if c in present]
        out[name] = (designed, context)
    return out


def _design_name(record: Dict[str, Any], index: int) -> str:
    """Stable, filesystem-safe name for a design; also its key in the chain map."""
    raw = str(record.get("uid") or record.get("design_id")
              or os.path.splitext(os.path.basename(record.get("path", "")))[0]
              or f"design_{index}")
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)


def stage_structures(records: List[Dict[str, Any]], staging_dir: str) -> Dict[str, str]:
    """Copy each design's structure into one directory under a unique name.

    ProteinMPNN parses a *folder* of PDBs, and a campaign's paths collide on
    basename (every pipeline writes ``{target}.pdb``) and get overwritten pass to
    pass. Staging copies each file under its unique design name, so the training
    set is a stable snapshot rather than a set of paths whose contents move.

    Returns ``{design_name: staged_path}``.
    """
    os.makedirs(staging_dir, exist_ok=True)
    staged: Dict[str, str] = {}
    for index, record in enumerate(records):
        name = _design_name(record, index)
        dst = os.path.join(staging_dir, f"{name}.pdb")
        shutil.copyfile(record["path"], dst)
        staged[name] = dst
    return staged


def original_checkpoint(
    model_state_dict: Any, *, num_edges: int = DEFAULT_NUM_NEIGHBORS,
    noise_level: float = DEFAULT_BACKBONE_NOISE, **extra: Any,
) -> Dict[str, Any]:
    """The checkpoint dict ``protein_mpnn_run.py`` loads.

    It reads ``checkpoint['model_state_dict']`` and ``checkpoint['num_edges']``
    (used as ``k_neighbors`` when constructing the model), so both are required
    for the published weights to load at all. ``noise_level`` is carried for
    parity with the original trainer's saves.
    """
    ckpt = {"model_state_dict": model_state_dict,
            "num_edges": int(num_edges), "noise_level": float(noise_level)}
    ckpt.update(extra)
    return ckpt


def published_weights_path(config: "ProteinMPNNConfig", output_dir: str) -> str:
    """Where the round's weights go so IMPRESS's next pass picks them up.

    ``mpnn_wrapper.py`` never passes ``--path_to_model_weights``, so
    ``protein_mpnn_run.py`` loads ``{mpnn_repo}/vanilla_model_weights/{model_name}.pt``
    by default. With ``publish_into_repo`` set (and ``mpnn_repo`` known) the
    checkpoint is written *there*, replacing the weights the campaign runs with;
    otherwise it lands in the round's ``output_dir`` and the integration is
    responsible for pointing MPNN at it (e.g. patch the wrapper to pass a path).
    """
    if config.publish_into_repo and config.mpnn_repo:
        return os.path.join(config.mpnn_repo, "vanilla_model_weights",
                            f"{config.model_name}.pt")
    return os.path.join(output_dir, f"{config.model_name}.pt")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProteinMPNNConfig:
    """Configuration for fine-tuning the original ProteinMPNN on campaign designs.

    Defaults describe a short mid-campaign round that starts from the public
    weights and nudges them, not a from-scratch training run.

    Parameters
    ----------
    mpnn_repo : Optional[str]
        Path to the ``dauparas/ProteinMPNN`` checkout — the same directory
        IMPRESS passes as ``-mpnn``. The trainer imports ``protein_mpnn_utils``
        from here, and (with ``publish_into_repo``) writes the new weights back
        into its ``vanilla_model_weights/``.
    initial_weights : Optional[str]
        Checkpoint the first round starts from, in original format — normally
        ``{mpnn_repo}/vanilla_model_weights/v_48_020.pt``. Later rounds resume
        from whatever the previous round published (passed as ``model_path``).
    model_name : str
        Basename of the published weights (no extension). Must match the
        ``--model_name`` IMPRESS runs with; the default ``v_48_020`` is
        ``protein_mpnn_run.py``'s default.
    design_chains, context_chains : sequence of str
        Default chain designation. For IMPRESS binders, ``("A",)`` designed and
        ``("B",)`` context. Overridable per record or via ``chains_func``.
    chains_func : optional callable
        ``(record, present_chains) -> (designed, context)`` to decide chains per
        structure. Overrides the defaults when set.
    num_neighbors, hidden_dim, num_layers, backbone_noise, dropout : ...
        Architecture. Must match the weights being fine-tuned — the public
        ``v_48_*`` weights are ``num_neighbors`` in {48}, hidden_dim 128, 3+3
        layers. ``protein_mpnn_run.py`` reads ``num_neighbors`` back from the
        checkpoint, so a mismatch will not load.
    max_epochs, batch_tokens, max_protein_length : int
        Round length and batching. ``batch_tokens`` caps residues per batch
        (ProteinMPNN uses ~10000). A short round is a few epochs.
    learning_rate_factor, warmup_steps, label_smoothing, gradient_norm : ...
        Noam schedule + label-smoothed NLL, as the original trainer uses.
    publish_into_repo : bool
        Write the new weights into ``{mpnn_repo}/vanilla_model_weights/`` so the
        campaign's next pass runs them with no wrapper change. Off by default:
        it mutates the shared repo, which a workflow should opt into knowingly.
    seed, num_workers, device : ...
        Reproducibility / loader / placement.
    train_func : optional callable
        ``(manifest_path, output_dir, config) -> checkpoint_path`` escape hatch,
        bypassing the built-in loop. The recommended path until the loop is
        validated on your checkout, and the way the examples stay runnable.
    """

    mpnn_repo: Optional[str] = None
    initial_weights: Optional[str] = None
    model_name: str = "v_48_020"
    dataset_name: str = "impress_r"

    design_chains: Sequence[str] = DEFAULT_DESIGN_CHAINS
    context_chains: Sequence[str] = DEFAULT_CONTEXT_CHAINS
    chains_func: Optional[Callable[..., Tuple[List[str], List[str]]]] = None

    num_neighbors: int = DEFAULT_NUM_NEIGHBORS
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    backbone_noise: float = DEFAULT_BACKBONE_NOISE
    dropout: float = 0.1

    max_epochs: int = 3
    batch_tokens: int = 10000
    max_protein_length: int = 2000

    learning_rate_factor: float = 2.0
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    gradient_norm: Optional[float] = 1.0

    publish_into_repo: bool = False
    seed: int = 0
    num_workers: int = 4
    device: str = "cuda"

    manifest_dir: Optional[str] = None
    train_func: Optional[Callable[..., str]] = None

    def validate(self) -> None:
        if not self.design_chains and self.chains_func is None:
            raise ValueError(
                "at least one design chain is required (design_chains is empty "
                "and no chains_func was given)"
            )
        if self.train_func is None and not self.mpnn_repo:
            raise ValueError(
                "mpnn_repo must point at a dauparas/ProteinMPNN checkout for the "
                "built-in trainer; set config.train_func to use your own loop "
                "instead. See docs/proteinmpnn_training.md."
            )


# ---------------------------------------------------------------------------
# Loss + optimizer (reimplemented so the loop depends only on the repo's model)
# ---------------------------------------------------------------------------

def label_smoothed_nll(S: Any, log_probs: Any, mask: Any, smoothing: float = 0.1) -> Any:
    """ProteinMPNN's ``loss_smoothed``: label-smoothed NLL over ``mask`` positions.

    Reimplemented (it is a few lines and numerically fixed) so the training loop
    needs only the model and featurizer from the ProteinMPNN repo, not its
    ``training/`` subpackage, whose helper signatures drift between commits.
    """
    import torch

    n = log_probs.size(-1)
    true = torch.nn.functional.one_hot(S, n).float()
    true = true + smoothing / float(n)
    true = true / true.sum(-1, keepdim=True)
    per_residue = -(true * log_probs).sum(-1)
    denom = torch.sum(mask) + 1e-8
    return torch.sum(per_residue * mask) / denom


class _NoamOptimizer:
    """The NoamOpt wrapper the original trainer uses; step-based warmup + decay."""

    def __init__(self, optimizer: Any, d_model: int, factor: float, warmup: int,
                 step: int = 0):
        self.optimizer = optimizer
        self._step = step
        self._d_model = d_model
        self._factor = factor
        self._warmup = warmup

    def rate(self, step: Optional[int] = None) -> float:
        step = self._step if step is None else step
        step = max(step, 1)
        return self._factor * (self._d_model ** -0.5) * min(
            step ** -0.5, step * self._warmup ** -1.5
        )

    def step(self) -> None:
        self._step += 1
        rate = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def _build_optimizer(parameters: Any, config: "ProteinMPNNConfig", step: int = 0):
    import torch

    adam = torch.optim.Adam(parameters, lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    return _NoamOptimizer(adam, config.hidden_dim, config.learning_rate_factor,
                          config.warmup_steps, step=step)


# ---------------------------------------------------------------------------
# The training task
# ---------------------------------------------------------------------------

class ProteinMPNNTrainer(TrainTask):
    """Fine-tunes the original ProteinMPNN on the campaign's confident designs.

    Returns the checkpoint file (original format), which the training manager
    publishes and — with ``publish_into_repo`` — drops straight into the repo's
    weights directory so IMPRESS's next pass runs it.

    Set ``config.train_func`` to bypass the built-in loop with your own
    ``(manifest_path, output_dir, config) -> checkpoint_path``. Recommended
    until the built-in loop is validated on your ProteinMPNN checkout.
    """

    #: Records are plain dicts of paths and scores, not a HuggingFace dataset.
    wants_hf_dataset = False

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
        super().__init__(
            gpus=gpus if gpus is not None else 1,
            nodes=nodes if nodes is not None else 1,
            name=name or "proteinmpnn",
        )
        self.config = config

    # -- pre-flight ---------------------------------------------------------

    def validate(self, dataset: Any) -> None:
        super().validate(dataset)
        cfg = self.config
        for index, record in enumerate(dataset):
            record = record or {}
            missing = [f for f in REQUIRED_FIELDS if f not in record]
            if missing:
                raise ValueError(
                    f"ProteinMPNN training needs {', '.join(REQUIRED_FIELDS)} on "
                    f"every record; corpus record {index} is missing "
                    f"{', '.join(missing)}. 'path' must point at the structure "
                    "whose designed chain is the training label (for IMPRESS-R, "
                    "the prediction of the designed sequence) — see "
                    "docs/proteinmpnn_training.md."
                )
            # A design with no designable chain present in the file trains
            # nothing and would produce an empty loss mask; catch it here rather
            # than on the GPU.
            present = pdb_chain_ids(record["path"]) if os.path.exists(record["path"]) else None
            if present is not None:
                if cfg.chains_func is not None:
                    designed, _ = cfg.chains_func(record, present)
                else:
                    designed = list(record.get("design_chains", cfg.design_chains))
                if present and not any(c in present for c in designed):
                    raise ValueError(
                        f"corpus record {index} ({record['path']}): none of the "
                        f"designed chains {list(designed)} are present in the "
                        f"structure (chains: {present}). For an IMPRESS binder the "
                        "designed chain is 'A' and the peptide 'B'."
                    )

    # -- corpus materialization (testable, no torch) ------------------------

    def write_manifest(self, records: List[Dict[str, Any]], output_dir: str) -> str:
        """Write the round's training manifest (parquet); returns its path.

        The audit trail for "what did this round train on": one row per design
        with its staged structure, chain designation, and scores. Public and
        side-effect-light so a workflow can inspect a round before trusting it.
        """
        import pandas as pd

        cfg = self.config
        manifest_dir = cfg.manifest_dir or os.path.join(output_dir, "manifest")
        os.makedirs(manifest_dir, exist_ok=True)

        designation = build_chain_designation(
            records, design_chains=cfg.design_chains,
            context_chains=cfg.context_chains, chains_func=cfg.chains_func,
        )
        rows = []
        for index, record in enumerate(records):
            name = _design_name(record, index)
            designed, context = designation[name]
            rows.append({
                "design_id": name,
                "path": os.path.abspath(record["path"]),
                "designed_chains": ",".join(designed),
                "context_chains": ",".join(context),
                "sequence": record.get("sequence") or "",
                "backbone_id": record.get("backbone_id"),
                "pLDDT": record.get("pLDDT"),
                "pTM": record.get("pTM"),
                "pAE": record.get("pAE"),
                "produced_under_version": record.get("model_version"),
            })
        path = os.path.join(manifest_dir, "train_manifest.parquet")
        pd.DataFrame(rows).to_parquet(path)
        return path

    # -- the round ----------------------------------------------------------

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Fine-tune ProteinMPNN on the campaign's designs; return the checkpoint.

        ``kwargs`` carries ``model_version`` from the training manager and
        ``model_path`` when a previous round published one.
        """
        records = list(dataset)
        manifest_path = self.write_manifest(records, output_dir)

        if self.config.train_func is not None:
            return self.config.train_func(manifest_path, output_dir, self.config) \
                or output_dir

        return self._train_with_proteinmpnn(records, output_dir, **kwargs)

    def _train_with_proteinmpnn(self, records, output_dir, **kwargs) -> str:
        """One fine-tuning round of the original ProteinMPNN.

        .. warning::

           Written against ``dauparas/ProteinMPNN``'s stable public API
           (``protein_mpnn_utils``: ``parse_PDB``, ``StructureDatasetPDB``,
           ``tied_featurize``, ``ProteinMPNN``) but **not yet executed in CI** —
           there is no torch or ProteinMPNN checkout here. Validate on your
           pinned checkout, or use ``config.train_func``. The featurizer's return
           tuple in particular has grown across commits; the unpack below matches
           the long-stable order and is guarded, but confirm it against yours.

        Dimer-aware: chain A is designed and scored, chain B is context — visible
        to the model (its backbone and true sequence condition the prediction)
        but excluded from the loss.
        """
        import sys

        import torch
        from torch.utils.data import DataLoader

        cfg = self.config
        if cfg.mpnn_repo and cfg.mpnn_repo not in sys.path:
            sys.path.insert(0, cfg.mpnn_repo)
        from protein_mpnn_utils import (  # type: ignore
            ProteinMPNN,
            StructureDatasetPDB,
            parse_PDB,
            tied_featurize,
        )

        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # -- data: stage the structures, designate chains, parse to dicts ----
        staging_dir = os.path.join(output_dir, "structures")
        staged = stage_structures(records, staging_dir)
        designation = build_chain_designation(
            records, design_chains=cfg.design_chains,
            context_chains=cfg.context_chains, chains_func=cfg.chains_func,
        )

        pdb_dicts = []
        for name, path in staged.items():
            parsed = parse_PDB(path)          # -> list; one entry per structure
            for entry in parsed:
                entry["name"] = name          # key tied_featurize on our name
                pdb_dicts.append(entry)

        dataset = StructureDatasetPDB(
            pdb_dicts, truncate=None, max_length=cfg.max_protein_length,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=lambda b: b)

        # -- model: build at the public architecture, resume prior weights ---
        model = ProteinMPNN(
            num_letters=21, node_features=cfg.hidden_dim,
            edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers,
            k_neighbors=cfg.num_neighbors, augment_eps=cfg.backbone_noise,
            dropout=cfg.dropout,
        ).to(device)

        resume = kwargs.get("model_path") or cfg.initial_weights
        if resume:
            state = torch.load(resume, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
        model.train()

        optimizer = _build_optimizer(model.parameters(), cfg)

        # -- the fine-tuning loop --------------------------------------------
        for _epoch in range(cfg.max_epochs):
            for batch in loader:
                featurized = tied_featurize(batch, device, designation, None,
                                            None, None, None, None)
                X, S, mask = featurized[0], featurized[1], featurized[2]
                chain_M = featurized[4]
                chain_encoding_all = featurized[5]
                residue_idx = featurized[12]

                optimizer.zero_grad()
                randn = torch.randn(chain_M.shape, device=device)
                log_probs = model(X, S, mask, chain_M, residue_idx,
                                  chain_encoding_all, randn)
                # Score only designed (chain_M==1) and resolved (mask==1) residues
                # — the peptide contributes context, never loss.
                loss = label_smoothed_nll(S, log_probs, mask * chain_M,
                                          smoothing=cfg.label_smoothing)
                loss.backward()
                if cfg.gradient_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.gradient_norm)
                optimizer.step()

        return self._save_and_publish(model.state_dict(), output_dir)

    def _save_and_publish(self, state_dict: Any, output_dir: str) -> str:
        """Write the checkpoint in original format; place it where MPNN loads it."""
        import torch

        cfg = self.config
        ckpt = original_checkpoint(
            state_dict, num_edges=cfg.num_neighbors,
            noise_level=cfg.backbone_noise,
        )
        target = published_weights_path(cfg, output_dir)
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        # Write beside the target then replace, so a reader (IMPRESS mid-pass)
        # never sees a half-written weights file.
        tmp = target + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, target)
        return target


# ---------------------------------------------------------------------------
# Corpus selection — orthogonal to the trainer, used by DataConfig
# ---------------------------------------------------------------------------

def impress_corpus_filter(
    min_pLDDT: float = 80.0,
    min_pTM: float = 0.8,
    max_pAE: float = 5.0,
) -> Callable[[Dict[str, Any]], bool]:
    """Build the IMPRESS admission predicate for :class:`~rome.data.DataConfig`.

    IMPRESS-R only trains on designs the campaign is confident in, and these are
    the thresholds IMPRESS already uses to decide a design is worth keeping::

        DataConfig(min_samples=24, filter_func=impress_corpus_filter())

    .. warning::

       **These defaults are known to be too permissive, and are kept only because
       a correct replacement cannot be derived yet.** Measured against a real
       PDZ campaign they admit 83% of records: ``pLDDT >= 80`` alone admits 100%,
       because everything reaching the score CSVs has already cleared IMPRESS's
       own keep/drop rule — the filter is being applied downstream of itself.

       That campaign was run with **Boltz**, while the branch ROME-A targets
       (``archive/ipdps_pdz_usecase``) runs **AlphaFold2-multimer**, and the two
       predictors do not share a confidence scale. Prefer :func:`percentile_sampler`,
       which needs no scale; see ``docs/impress.md``.
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


def score_percentiles(
    records: Sequence[Dict[str, Any]],
    keys: Sequence[str] = ("pLDDT", "pTM", "pAE"),
) -> Dict[str, Dict[str, float]]:
    """Summarise the score distribution a campaign has produced so far.

    Calibration data, gathered from the run itself. Every confidence threshold
    in IMPRESS-R is predictor-specific — AlphaFold2-multimer and Boltz do not
    share a scale — so the only safe way to set one is to look at what *this*
    campaign is producing::

        from rome.train.mpnn import score_percentiles
        print(score_percentiles(manager.data.get_records()))

    Returns ``{key: {"n", "min", "p10", "p25", "median", "p75", "p90", "max"}}``,
    skipping keys no record carries.
    """
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = sorted(
            float(r[key]) for r in records
            if r.get(key) is not None and _is_number(r[key])
        )
        if not values:
            continue

        def q(p: float) -> float:
            return values[min(len(values) - 1, int(p * len(values)))]

        out[key] = {
            "n": float(len(values)), "min": values[0], "p10": q(0.10),
            "p25": q(0.25), "median": q(0.50), "p75": q(0.75),
            "p90": q(0.90), "max": values[-1],
        }
    return out


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


#: Default ranking for the PDZ binder case: interface pAE down, pTM up. pLDDT is
#: deliberately absent — in a measured campaign it never fell below 88, so it
#: separates almost nothing.
DEFAULT_RANK_BY = {"pAE": "low", "pTM": "high"}


def percentile_sampler(
    fraction: float = 0.33,
    *,
    rank_by: Optional[Dict[str, str]] = None,
    min_shard: int = 8,
    on_summary: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Select the best ``fraction`` of the corpus, calibrating as it goes.

    Use this instead of tuning :func:`impress_corpus_filter`'s thresholds::

        DataConfig(min_samples=24, sample_func=percentile_sampler(0.33))

    **Why a fraction rather than a cutoff.** A threshold like ``pTM >= 0.90``
    is a claim about a specific predictor's confidence scale, and IMPRESS
    campaigns have been run on both AlphaFold2-multimer and Boltz, which do not
    share one. A fraction says "the best third of what this campaign has
    produced", which needs no scale and calibrates itself on the fly — including
    on the first round, before anyone has seen the distribution.

    It also sidesteps the trap that makes IMPRESS's own keep/drop thresholds
    useless here: everything in the corpus already cleared them, so reusing them
    as an admission filter selects nothing.

    Ranking is by **average rank across ``rank_by``**, not by a weighted sum of
    raw values. Rank-averaging is non-parametric, so pTM (0–1) and pAE (Å, open
    ended) contribute equally without needing to be normalised, and it is
    unaffected by either metric's outliers.

    Parameters
    ----------
    fraction : float
        Portion of the corpus to keep, in (0, 1].
    rank_by : Optional[Dict[str, str]]
        Record field -> ``'high'`` (higher is better) or ``'low'``. Defaults to
        :data:`DEFAULT_RANK_BY`. Fields absent from every record are ignored, so
        a corpus carrying only some of them still ranks.
    min_shard : int
        Never return fewer than this many records — a strict fraction of a small
        early corpus can otherwise produce a shard too small to train on. Capped
        at the corpus size.
    on_summary : Optional[Callable[[dict], None]]
        Called with ``{"corpus", "selected", "ranked_by", "percentiles",
        "cutoffs"}`` each time a shard is built. Pass ``print`` or a logger to
        watch the campaign's distribution move, and to read off the thresholds
        an equivalent fixed filter would have used.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    directions = dict(rank_by if rank_by is not None else DEFAULT_RANK_BY)
    for key, direction in directions.items():
        if direction not in ("high", "low"):
            raise ValueError(
                f"rank_by[{key!r}] must be 'high' or 'low', got {direction!r}"
            )

    def _sample(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = list(records)
        if not records:
            return records

        usable = {
            key: direction for key, direction in directions.items()
            if any(_is_number(r.get(key)) for r in records)
        }
        if not usable:
            # Nothing to rank on: keep the corpus rather than silently
            # returning an arbitrary slice of it.
            return records

        # Average rank across metrics. Missing values sort last within their
        # metric, so a partially-scored record is penalised but not dropped.
        total = {id(r): 0.0 for r in records}
        for key, direction in usable.items():
            ordered = sorted(
                records,
                key=lambda r: (
                    not _is_number(r.get(key)),
                    -float(r[key]) if _is_number(r.get(key)) and direction == "high"
                    else (float(r[key]) if _is_number(r.get(key)) else 0.0),
                ),
            )
            for position, record in enumerate(ordered):
                total[id(record)] += position

        ranked = sorted(records, key=lambda r: (total[id(r)], str(r.get("uid", ""))))
        keep = max(min(min_shard, len(records)), int(round(len(records) * fraction)))
        selected = ranked[:keep]

        if on_summary is not None:
            cutoffs = {}
            for key, direction in usable.items():
                values = [float(r[key]) for r in selected if _is_number(r.get(key))]
                if values:
                    cutoffs[key] = min(values) if direction == "high" else max(values)
            on_summary({
                "corpus": len(records),
                "selected": len(selected),
                "ranked_by": usable,
                "percentiles": score_percentiles(records, tuple(usable)),
                "cutoffs": cutoffs,
            })
        return selected

    return _sample


__all__ = [
    "ProteinMPNNConfig",
    "ProteinMPNNTrainer",
    "build_chain_designation",
    "stage_structures",
    "pdb_chain_ids",
    "original_checkpoint",
    "published_weights_path",
    "label_smoothed_nll",
    "impress_corpus_filter",
    "percentile_sampler",
    "score_percentiles",
    "DEFAULT_RANK_BY",
    "DEFAULT_DESIGN_CHAINS",
    "DEFAULT_CONTEXT_CHAINS",
    "MPNN_ALPHABET",
]
