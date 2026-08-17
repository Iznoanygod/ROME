# Training ProteinMPNN for IMPRESS-R

What foundry's ProteinMPNN trainer actually consumes, what IMPRESS has to hand it,
and what is still missing.

Everything below is read off
[`RosettaCommons/foundry@production`](https://github.com/RosettaCommons/foundry/tree/production)
(`models/mpnn/`) and its `atomworks>=2.1.1` dependency. The repo's `models/mpnn/README.md`
training section is still a stub — `models/mpnn/src/mpnn/train.py` is the real
reference, and it is a complete from-scratch pretraining script.

---

## 1. The headline: it is not a sequence dataset

ProteinMPNN is **inverse folding** — structure in, sequence out — so the label
*is* the sequence embedded in the structure file. There is no sequence column,
and no place to put one.

A foundry training set is a **pandas DataFrame (parquet) of metadata rows, one
row per training example, each row pointing at a structure file on disk**:

```python
StructuralDatasetWrapper(
    dataset=PandasDataset(data=train_df, id_column="example_id", filters=[...]),
    dataset_parser=GenericDFParser(
        example_id_colname="example_id",
        path_colname="path",
        assembly_id_colname="assembly_id",
    ),
    transform=build_mpnn_transform_pipeline(model_type="protein_mpnn", is_inference=False),
    cif_parser_args={**STANDARD_PARSER_ARGS, "load_from_cache": True, ...},
)
```

atomworks parses the file at `path`, the transform pipeline featurizes it, and
`LabelSmoothedNLLLoss` scores predicted residue identity against the residues in
that file. Metrics are NLL and sequence recovery.

**Consequence for IMPRESS-R:** a training example is one structure file whose
coordinates are what the model conditions on and whose sequence is what the model
must reproduce. Our current `write_shard()` in `rome/train/mpnn.py` — which writes
`{sequence, pdb_path, pLDDT, ...}` rows to parquet — is the wrong shape, and the
`MPNNTrainer(train_data=..., output_dir=..., checkpoint=...)` call it makes does
not exist. See §7.

---

## 2. Required dataframe columns

Split by what consumes them.

| Column | Consumed by | IMPRESS-R value |
|---|---|---|
| `example_id` | `GenericDFParser`, `PandasDataset` id | `generate_example_id(["impress_r"], design_id, "1", ["A_1"])` |
| `path` | `GenericDFParser` | absolute path to the design's structure file |
| `assembly_id` | `GenericDFParser` | `"1"` |
| `n_non_atomized_tokens` | `PaddedTokenBudgetBatchSampler`, filters | residue count — for the PDZ case the designed chain **plus** the 10-mer peptide |
| `cluster` | `calculate_weights_for_pdb_dataset_df` | `backbone_id` — see below |
| `n_prot` | AF3 weighting | `2` for the PDZ binder case (designed chain + peptide); `1` for a monomer |
| `n_peptide` | AF3 weighting | `1` if ≤20 residues else `0` (`PEPTIDE_MAX_RESIDUES = 20`) |
| `n_nuc`, `n_ligand` | AF3 weighting | `0`, `0` |
| `q_pn_unit_is_loi` | AF3 weighting | `0` |
| `resolution`, `method`, `deposition_date` | **stock filters only** | not applicable — drop those filters |

Two traps worth knowing before you hit them:

- **`n_peptide` is required but not asserted.** `calculate_af3_example_weights`
  asserts on `["n_prot", "n_nuc", "n_ligand", "cluster_size"]` and then reads
  `df["n_peptide"]` unguarded. Omitting it gives a bare `KeyError`, not the
  helpful assertion message.
- **The stock `MPNN_FILTERS` are PDB-specific.** They reference `resolution`,
  `method`, and `deposition_date`, which a design campaign has no analogue for.
  `PandasDataset` filters are pandas query strings, so a missing column is a hard
  error. Pass campaign-appropriate filters instead — the only stock one worth
  keeping is `n_non_atomized_tokens >= 30` — and note `n_prot == 1` is *not* one
  of them for the PDZ case, where it would reject every example (§5).

**Clustering matters more than it looks.** The AF3 weighting is
`w ∝ (β / N_cluster) · (a_prot·n_prot + …)`, so cluster assignment is the only
thing stopping one backbone that produced 200 accepted designs from dominating
the round. Clustering by `backbone_id` gives each backbone equal total weight
regardless of how many of its designs passed the filter — which is almost
certainly what you want in a campaign where some backbones are far easier than
others. Cluster sizes must be > 0 and non-null (asserted).

---

## 3. The real design decision: which (structure, sequence) pair?

IMPRESS produces, per design: backbone `B` → MPNN → sequence `S` → AF2 →
predicted structure `P`, plus pLDDT/pTM/pAE and TM-score(P, B).

That gives two legitimate ways to build a training example, and they teach
different things:

**Option A — `(B, S)`: "on this target backbone, produce S".** Exactly the
conditioning distribution seen at inference, so it is the more on-distribution
choice. This is classic expert iteration / rejection-sampling self-training: the
signal comes entirely from the admission filter, since you keep only the designs
that folded well. Two costs: `B`'s file carries the *original* sequence, so you
must thread `S` onto `B`'s coordinates and write a new structure file; and
training a model purely on its own accepted outputs is the setup where mode
collapse shows up.

**Option B — `(P, S)`: "given the structure this sequence actually folds to,
produce S".** This is AF2 distillation, and it adds (structure, sequence) pairs
that are not in the PDB at all. atomworks has explicit precedent for it — the
`generate_example_id` docstring gives `{['af2_distillation']}{6vyb}{1}{['A_1']}`
as a worked example.

**Recommendation: Option B, because it is free.** The ColabFold output PDB
*already is* a valid training example — its coordinates are `P` and its sequence
is `S`, with no threading step and no new files to write. The pLDDT/pTM filter is
also a statement about precisely that structure, so filter and training example
refer to the same object. Option A needs a threading step built before it can be
evaluated at all.

Worth stating in the paper as a deliberate choice rather than an implementation
detail, since it is the substantive experimental decision in IMPRESS-R.

---

## 4. What IMPRESS has to emit

Per accepted design, ROME-A's data manager needs:

```python
rome.add_training_data(
    path=<absolute path to that design's AF2 PDB>,   # the training example itself
    sequence=<designed sequence>,                     # for n_non_atomized_tokens + audit
    backbone_id=<parent backbone>,                    # -> cluster
    pLDDT=..., pTM=..., pAE=...,                      # -> admission filter
    score=pLDDT,
)
```

Most of this IMPRESS already has. Three things need attention:

1. **The per-design PDB path must be real, not reconstructed.**
   `oldrome/protein/tasks.py::_parse_extract_csv` currently synthesizes
   `os.path.join(af_output_dir, f"{ident}.pdb")` from the extractor's CSV `ID`
   column. ColabFold does not write that filename — it writes
   `<name>_unrelaxed_rank_001_alphafold2_ptm_model_N_seed_000.pdb`. The path has
   to be resolved by globbing the actual output, or the whole corpus points at
   files that do not exist.

2. **Pick one model per design and be consistent.** `run_fold.sh` runs
   `--num-models 5`. Train on `rank_001` only; mixing ranks silently mixes
   confidence levels within one corpus.

3. **Residue count.** Cheap from the sequence, but it drives the token-budget
   batch sampler, so it has to be right or batching misbehaves.

---

## 5. Gaps and risks

- **Catastrophic forgetting is the main scientific risk.** The reference script
  trains on PDB-scale data with AF3 example weighting. Fine-tuning on a few
  hundred self-generated designs, with no PDB data mixed in, will drift the model
  toward the campaign and away from general inverse folding. The standard
  mitigation is to mix a slice of the original training distribution into each
  round — but **foundry has not released the dataframes** ("we are working to
  release the dataframes used for retraining"), so with released artifacts alone
  that mitigation is not currently available. Options: keep rounds very short and
  the learning rate very low; hold out a fixed validation set from an earlier
  campaign and watch sequence recovery per round; or ask IPD for the split
  dataframes.
- **Binder designs are multi-chain — and the PDZ campaign is one.** Settled from
  IMPRESS's own source on the branch ROME-A targets, not from campaign output:
  `protein_binding.py` writes the prediction input as two chains, a designed
  `>pdz` and a constant `>pep` of `EGYQDYEPEA`, and `plddt_extract_pipeline.py`
  averages pAE over exactly the cross terms with those last 10 residues. So
  `n_prot == 2`, `avg_pae` is an *interface* pAE, and **the stock `n_prot == 1`
  filter rejects every example** — it has to be relaxed and the alphas set for a
  two-chain complex before a shard writer produces anything usable.

  The asymmetry matters: only chain A is designed. Chain B is the same 10-mer in
  every example, so it is fixed context rather than a second training target, and
  weighting it as a peptide chain (`n_peptide`) is closer to the truth than
  counting two protein chains.
- **The re-implementation is explicitly unstable.** foundry's README carries both
  an API-instability warning and a benchmarking warning ("please use the old
  repositories … until the API and public weights stabilize"). Pin a commit.
- **`.cif` is preferred over `.pdb`.** atomworks parses both (`file_type` is
  inferred), but the docstring says "`.cif` files are strongly recommended".
  Converting ColabFold output to CIF once, at corpus-admission time, is probably
  worth it.

---

## 6. Checkpoint lifecycle — this part fits ROME-A cleanly

The handoff between foundry's trainer and its inference engine is directly
compatible with ROME-A's publish/reload protocol.

**Output.** `FabricTrainer.save_checkpoint` writes
`<output_dir>/ckpt/epoch-NNNN.ckpt` via `fabric.save(path, self.state)`, so the
file is a dict with a `"model"` key.

**Inference.** `MPNNInferenceEngine(checkpoint_path=..., is_legacy_weights=False)`
does `checkpoint["model"]` → `load_state_dict(strict=True)`. So a checkpoint the
trainer just wrote is loadable by the stream with no conversion. Note it requires
a **file**, not a directory (`ckpt_path.is_file()` is validated), so ROME-A must
publish the `.ckpt` file path rather than the round's output directory.

**Round N > 1.** Resume with
`CheckpointConfig(path=<previous ckpt dir or file>, reset_optimizer=False)`,
which keeps optimizer and scheduler state across rounds — important, since the
Noam schedule is step-based and resetting it every round would re-warm-up each
time.

**Round 1 bootstrap from released weights.** The public ProteinMPNN weights
(`proteinmpnn_v_48_020.pt`) are legacy-format and need `load_legacy_weights`,
which renames parameters, drops an unused atom-type embedding, permutes the
pairwise-distance embedding, and reorders the token vocabulary (legacy is
alphabetical by 1-letter code, new is alphabetical by 3-letter code). That
function loads into a live model, not into a checkpoint file, so bootstrapping is
a one-time conversion:

```python
model = ProteinMPNN()
load_legacy_weights(model, "proteinmpnn_v_48_020.pt")
torch.save({"model": model.state_dict()}, "proteinmpnn_v48_020_converted.ckpt")
```

then load it with `reset_optimizer=True` (there is no optimizer state to
restore). `load_checkpoint` only touches `ckpt["model"]` in that mode, so a
one-key checkpoint is fine.

**Gotcha:** `get_latest_checkpoint` returns `sorted(dir.iterdir())[-1]` — every
file in `ckpt/`, not just `*.ckpt`. A stray log or temp file in that directory
silently becomes "the latest checkpoint".

---

## 7. Consequences for `rome/train/mpnn.py`

The current implementation was written against a guessed API and is wrong in
three ways:

1. `MPNNTrainer(train_data=..., output_dir=..., checkpoint=...)` / `.fit()` does
   not exist. The real construction is
   `MPNNTrainer(model_type=..., accelerator=..., devices_per_node=..., max_epochs=...,
   output_dir=...)` followed by `initialize_or_update_trainer_state({"train_cfg": ...})`,
   `fabric.launch()`, `construct_model()`, `construct_optimizer()`,
   `construct_scheduler()`, then `fit(train_loader=..., val_loaders=..., ckpt_config=...)`.
   The optimizer and LR schedule are supplied by the caller as an OmegaConf
   `train_cfg` with `_target_` entries, not by the trainer.
2. The shard is the wrong shape — see §1. It must be a metadata dataframe with
   the §2 columns, plus a DataLoader built from `StructuralDatasetWrapper`,
   `PaddedTokenBudgetBatchSampler` and `TokenBudgetAwareFeatureCollator`.
3. It returns the output *directory*, but the inference engine needs the
   `.ckpt` **file**.

Also note the reference script's scale is from-scratch pretraining
(`max_epochs=500`, `n_examples_per_epoch=20000`, batch budget 10000 tokens). A
mid-campaign ROME-A round wants `n_examples_per_epoch` ≈ corpus size,
`max_epochs` in the single digits, and `checkpoint_every_n_epochs=1` so a
checkpoint exists to publish as soon as the round ends.

---

## 8. Open questions

- ~~Monomer designs or binders?~~ **Answered: binders**, from IMPRESS's own
  source rather than campaign output, so it holds for the branch ROME-A targets:
  a designed PDZ chain plus a constant 10-mer peptide. `n_prot == 2` and the
  stock filter must be relaxed (§5).
- Corpus size is the binding constraint, not compute. IMPRESS folds **one**
  sequence per pipeline per pass regardless of `num_seqs` (it picks a single rank
  from MPNN's 10 and scores only that), so a 70-pipeline campaign yields ~70
  labelled designs per pass and a few hundred over a whole run. A round trains on
  **tens of examples**. Per-round epochs and learning rate matter far more than
  throughput, and it sharpens the drift question below.
- Would scoring more of MPNN's sequences per pass be worth proposing upstream?
  Nine of ten are discarded unscored. That is the cheapest available lever on
  corpus size, but it costs one structure prediction per extra sequence, so it
  trades directly against campaign throughput.
- Option A or Option B pairing (§3)? B is free; A needs a threading step built.
- Is there access to IPD's split dataframes for mixing in PDB data, or does
  IMPRESS-R accept pure self-training and measure the drift (§5)?
- Which foundry commit to pin, given the stated API instability?
