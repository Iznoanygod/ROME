# Fine-tuning ProteinMPNN for IMPRESS-R

What the trainer does, what data it needs, and what still has to be validated on
a real checkout. This replaces an earlier version written against foundry's
ProteinMPNN re-implementation — that was the wrong target, for the reason in §1.

---

## 1. Which ProteinMPNN — and why it matters

IMPRESS runs the **original `dauparas/ProteinMPNN`**. Its `mpnn_wrapper.py`
shells out to `protein_mpnn_run.py` with the original CLI
(`--jsonl_path`, `--chain_id_jsonl`, `--fixed_positions_jsonl`, …) and its
`helper_scripts/`, and the setup clones that repo directly:

```
git clone https://github.com/dauparas/ProteinMPNN.git
```

So the trainer fine-tunes **that** model and writes a checkpoint in **that**
format. This is not a detail — a checkpoint has to load into the model IMPRESS
runs at inference:

* the original loads `torch.load(path)['model_state_dict']` and reads
  `checkpoint['num_edges']` to size the neighbour graph;
* foundry's re-implementation uses a different `{"model": ...}` layout, different
  parameter names, and a reordered amino-acid vocabulary.

A foundry checkpoint will not load into `protein_mpnn_run.py`, and there is no
foundry→original converter. Targeting the original removes the mismatch entirely.

(PyRosetta is in the IMPRESS stack, but for FastRelax and pLDDT extraction —
`plddt_extract_pipeline.py` — not for sequence design. The design model is
vanilla ProteinMPNN.)

## 2. The training example is a *dimer*, scored on the designed chain

An IMPRESS protein-binding design is a complex, not a monomer:
`protein_binding.py` writes the prediction input as two chains — a designed
`>pdz` and a constant `>pep` of `EGYQDYEPEA` — and the AF/Boltz prediction of
that pair (coordinates + the designed sequence in chain A) is the training
example.

Fine-tuning has to respect that. ProteinMPNN's chain mask expresses exactly the
right thing:

* **chain A — designed.** Predicted and scored: this is the sequence recovery
  the campaign is optimising.
* **chain B — context.** Its backbone *and* its known sequence condition the
  prediction of chain A, but it is excluded from the loss. Learning to design a
  binder means learning it *in the presence of* the target, which is what makes
  the peptide context rather than a second target.

`build_chain_designation` produces `{design_name: (["A"], ["B"])}`, which is what
the original repo's `tied_featurize` consumes to build that mask. The loss is
computed over `mask * chain_M` — resolved residues of the designed chain only.

Defaults are `design_chains=("A",)`, `context_chains=("B",)`. A monomer campaign
sets `context_chains=()`; a per-structure rule can be supplied as
`chains_func(record, present_chains) -> (designed, context)`. Chains named but
absent from a file are dropped, and a design with no designable chain present is
rejected in `validate` rather than producing an empty loss on the GPU.

## 3. What data a round needs

Per accepted design, a corpus record needs:

```python
manager.add_training_data(
    path=<the design's predicted structure, chain A designed, chain B context>,
    sequence=<designed sequence>,     # optional: manifest + sizing only
    backbone_id=<parent pipeline>,    # audit
    pLDDT=..., pTM=..., pAE=...,       # selection (see docs/impress.md)
)
```

Only `path` is required — ProteinMPNN reads the label (the designed chain's
residues) out of the structure itself; there is no separate sequence label. Two
things the integration must get right, both from `docs/impress.md`:

1. **The path must be a stable snapshot.** IMPRESS keys its prediction output by
   pipeline, not by pass, and `finalize()` deletes it, so a recorded path goes
   stale. The trainer defends against basename collisions by *staging* every
   structure into the round's own directory under a unique name
   (`stage_structures`), but the contribution step still has to record a path
   whose contents are correct at record time.
2. **Rank by pAE/pTM, not pLDDT** when selecting which designs enter the corpus —
   pLDDT barely moves. Prefer `percentile_sampler`; see `docs/impress.md` §9.

## 4. What a round produces, and where it goes

The round writes a checkpoint in the original format —
`{"model_state_dict": ..., "num_edges": 48, "noise_level": 0.2}` — via
`original_checkpoint`, the exact dict `protein_mpnn_run.py` loads.

Getting it *back into the campaign* is the seam that actually closes the loop.
`mpnn_wrapper.py` never passes `--path_to_model_weights`, so
`protein_mpnn_run.py` loads `{mpnn_repo}/vanilla_model_weights/{model_name}.pt`
by default. With `publish_into_repo=True` the trainer writes the new weights
*there* (atomically — temp file then `os.replace`, so a mid-pass reader never
sees a half-written file), replacing what the campaign runs with. The next pass
picks them up with no wrapper change. `model_name` must match the `--model_name`
IMPRESS runs (`v_48_020` by default).

Without `publish_into_repo`, the checkpoint lands in the round's `output_dir` and
pointing MPNN at it is the integration's job — e.g. patch the wrapper to pass
`--path_to_model_weights`.

## 5. Config that must match the weights

The public `v_48_*` weights fix the architecture, and `protein_mpnn_run.py`
hardcodes most of it (hidden_dim 128, 3+3 layers) while reading `k_neighbors`
back from the checkpoint's `num_edges`. So a fine-tune must keep
`num_neighbors=48`, `hidden_dim=128`, `num_layers=3`, or the published checkpoint
will not load. These are the `ProteinMPNNConfig` defaults; change them only if
you are also changing the weights IMPRESS runs.

A round is short by design: `max_epochs=3`, batching to `batch_tokens=10000`,
resuming from the previously published checkpoint (or `initial_weights` on round
one) with a Noam schedule, label-smoothed NLL — the original trainer's own recipe.

## 6. What is tested, and what is not

Tested here, with no torch or ProteinMPNN checkout (`tests/unit/test_mpnn_trainer_data.py`,
`tests/unit/test_train_tasks.py`):

* chain designation, including the dimer default and its overrides;
* structure staging under unique names;
* `validate` — required `path`, all records scanned, designed chain present;
* the manifest (the audit trail);
* the checkpoint format and the weight-publication path.

**Not** run in CI, and the one thing to validate on your allocation:
`_train_with_proteinmpnn`, the torch loop. It is written against the original
repo's stable public API — `parse_PDB`, `StructureDatasetPDB`, `tied_featurize`,
`ProteinMPNN` from `protein_mpnn_utils` — plus a reimplemented label-smoothed loss
and Noam optimiser so it depends only on the model and featurizer, not the
repo's `training/` subpackage (whose helper signatures drift between commits).
One thing to confirm against your checkout: `tied_featurize`'s return tuple has
grown over time; the unpack takes the long-stable positions and is guarded, but
check it. Until it is validated, set `config.train_func` to your own
`(manifest_path, output_dir, config) -> checkpoint_path` — that is the path the
examples run on, and the recommended way to bring up a real campaign one layer
at a time.

## 7. Open items

* **Drift.** Fine-tuning only on self-generated designs pulls the model toward
  the campaign. The standard mitigation mixes in a slice of the original PDB
  training distribution; the campaign does not provide one, so either hold out a
  validation set from an earlier campaign and watch sequence recovery per round,
  or keep rounds short and the learning rate low.
* **Pin the checkout.** The dauparas repo's `tied_featurize` and helper layout
  have changed over time. Pin a commit and validate the loop against it once.
