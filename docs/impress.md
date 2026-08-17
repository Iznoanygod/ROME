# Running IMPRESS (archive/ipdps_pdz_usecase)

Status: **both dummy examples working**, and IMPRESS's own test suite passes
(19 tests). One incompatibility had to be worked around; nothing else.

Branch: [`radical-collaboration/IMPRESS@archive/ipdps_pdz_usecase`](https://github.com/radical-collaboration/IMPRESS/tree/archive/ipdps_pdz_usecase)
(tip `baf42a8`, 2026-07-06).

---

## Install

```bash
git clone --branch archive/ipdps_pdz_usecase --single-branch \
  https://github.com/radical-collaboration/IMPRESS.git
cd IMPRESS
pip install --no-deps -e .
```

`--no-deps` is deliberate. `pyproject.toml` declares `radical.pilot`, but that is
only needed for the RADICAL execution backend — the dummy examples and the test
suite run without it, verified with `radical.pilot` absent from the environment.
Installing it pulls in a large stack you do not need to see these work.

The only other requirement is `radical-asyncflow`, which ROME-A already needs.

## The one incompatibility

The examples do not run as written against a current asyncflow:

```
ImportError: cannot import name 'ConcurrentExecutionBackend' from 'radical.asyncflow'
```

`ConcurrentExecutionBackend` exists **only in asyncflow 0.2.0**; it was renamed
to `LocalExecutionBackend` in 0.3.0, and the archived branch predates that. I
checked every published version:

| asyncflow | backends exported |
|---|---|
| 0.2.0 | `ConcurrentExecutionBackend`, `DaskExecutionBackend`, `NoopExecutionBackend`, `RadicalExecutionBackend` |
| 0.3.0 – 0.5.0 | `LocalExecutionBackend`, `NoopExecutionBackend` |

Two ways to run them, and for IMPRESS-R only one of them is viable:

1. **Current asyncflow (recommended).** Rename the backend in the example — two
   lines, an import and the construction:

   ```diff
   -from radical.asyncflow import ConcurrentExecutionBackend
   +from radical.asyncflow import LocalExecutionBackend
   ...
   -    execution_backend = await ConcurrentExecutionBackend(ThreadPoolExecutor())
   +    execution_backend = await LocalExecutionBackend(ThreadPoolExecutor())
   ```

   Nothing else needed: IMPRESS touches only `WorkflowEngine.create(backend=...)`
   and `flow.executable_task(...)`, both unchanged since 0.2.0.

2. **Pin `radical-asyncflow==0.2.0`** and run the examples untouched. Fine for
   reproducing the archive, but ROME-A needs 0.5.0 — its streams are submitted
   as asyncflow *service* tasks, which 0.2.0 has no concept of — so IMPRESS-R
   cannot take this path.

Since the branch is archived, the rename is best applied in a working copy
rather than pushed back.

## What was verified

```
IMPRESS test suite                19 passed
examples/dummy.py                 3 pipelines, s1 -> s2 -> s3 each, clean exit
examples/dummy_adaptive.py        3 pipelines through
                                  sequence_analysis -> fitness_evaluation ->
                                  adaptive step -> optimization_step, clean exit
```

`dummy_adaptive.py` also exercises child-pipeline spawning — one run produced
`Submitting child pipeline: p3_g2 from p3`, which then ran its own full cycle
and completed. Whether a child spawns is a coin flip in the example
(`random.random() >= 0.5` gates it), so it does not happen every run; the
machinery works when the gate opens.

---

## IMPRESS-R: where ROME-A attaches

Verified end to end in `examples/impress_r/adaptive_rome.py`, which runs a real
`ImpressManager` driving a real `ImpressBasePipeline` with a real
`rome.Manager`. Only the AlphaFold and ProteinMPNN executables are stubbed, so
it runs anywhere:

```bash
dragon -s examples/impress_r/adaptive_rome.py
```

```
[PIPELINE-P1] pass 1 | mpnn=proteinmpnn_v_48_020.pt
[PIPELINE-P1] corpus 4 (+4 this pass) | WAITING
[PIPELINE-P1] pass 2 | mpnn=proteinmpnn_v_48_020.pt
[PIPELINE-P1] ROME-A published v1
[PIPELINE-P1] pass 3 | mpnn=/tmp/impress_r_.../checkpoints/dummy/v1     <-- swapped
[PIPELINE-P1] corpus 12 (+6 this pass) | WAITING
...
[PIPELINE-P1] pass 10 | mpnn=/tmp/impress_r_.../checkpoints/dummy/v6
[PIPELINE-P1] ROME-A published v7
```

The campaign runs its passes with baseline weights, ROME-A publishes v1 after
pass 2, and **pass 3 onward runs MPNN with the campaign's own checkpoint**. The
per-pass acceptance count climbs (+4, +2, +6, +8) as the model improves, which
is the loop closing.

### `adaptive_fn` is the seam

In the protein-binding use case, `adaptive_decision(pipeline)` runs after the
pLDDT-extraction task of every pass, reads
`af_stats_{name}_pass_{n}.csv`, and decides which designs regressed. That is the
one point in the campaign that both *has* fresh scored designs and is *between*
passes, so both halves of ROME-A go there:

```python
async def adaptive_decision(pipeline):
    with open(f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv") as fd:
        for row in csv.DictReader(fd):                 # ID, avg_plddt, ptm, avg_pae
            design = row["ID"].split(".")[0]
            manager.add_training_data(                 # 1. contribute
                path=f"{pipeline.output_path_af}/{design}.pdb",
                sequence=pipeline.iter_seqs[design],
                backbone_id=pipeline.name,
                pLDDT=float(row["avg_plddt"]), pTM=float(row["ptm"]),
                pAE=float(row["avg_pae"]), score=float(row["avg_plddt"]))

    weights = manager.get_current_model()              # 2. collect
    if weights:
        pipeline.mpnn_weights = weights                #    next pass uses it

    ...                                                # IMPRESS's own logic,
                                                       # unchanged
```

`run()` never mentions ROME-A. The degradation logic that spawns child
pipelines is untouched.

Two details that fall out of the real code:

* **`pipeline.output_path_af/{design}.pdb` is the training example.** It is the
  AlphaFold prediction of the designed sequence — coordinates and sequence in
  one file, which is exactly what ProteinMPNN trains on, with no threading step.
  The real `adaptive_decision` already copies these files when migrating a
  backbone, so the path is known-good. See `docs/proteinmpnn_training.md`.
* **The score columns are `ID, avg_plddt, ptm, avg_pae`**, and IMPRESS's own
  criterion reads the last one: a *rising* `avg_pae` means degraded. Feed
  `avg_plddt` to ROME-A as the corpus score and let
  `impress_corpus_filter()` apply all three thresholds.

### Which workflow engine

Both work, and `rome.Manager` now supports either:

```python
rome.Manager()                        # builds its own engine at start(),
                                      # shuts it down at stop()
rome.Manager(impress_manager.flow)    # shares the campaign's engine
```

Giving ROME-A its own is the default in the example: its training rounds are
then scheduled independently of the campaign's tasks, which is what "ROME-A
handles its own task management" means in practice. Sharing puts rounds and
campaign tasks in one engine against one allocation — better when the two must
compete for the same fixed resources.

One wrinkle if you share: `ImpressManager` creates its engine *inside*
`start()`, and `start()` does not return until the whole campaign is done. So
`impress_manager.flow` does not exist when you would want to construct the
ROME-A manager. Either build the engine yourself first and hand it to both, or
let ROME-A build its own. Both paths are covered by
`tests/integration/test_impress_r.py`.

### Other seams worth knowing

* **`auto_register_task(local_task=True)` leaves the function alone** instead of
  wrapping it as an executable task. That is how a Python-level step — anything
  that should not become a shell command — lives inside a pipeline.
* **`PipelineSetup(kwargs={...})`** passes arbitrary configuration through to
  the pipeline constructor, which is how the example threads `base_path`
  through.


---

## What the real campaign data says

From a 70-target PDZ binder campaign: the `IMPRESS data` Drive folder
(`prod_in_70` inputs, `prod/p1-p16` results), plus two files measured directly —
all 176 `af_stats_*.csv` from the `p1-p16` group, and `sequences_indexed.csv`
covering all 70 targets. These correct several assumptions the example was
built on.

### Inputs: 70 pipelines, one structure each

`prod_in_70/` holds `p1_in` … `p70_in`, matching IMPRESS's `{name}_in`
convention. Each contains exactly **one PDB** — `p1_in/2ejy.pdb`, 129 KB. So a
campaign is 70 independent pipelines, each seeded with a single PDZ complex, not
a pool of backbones per pipeline.

### One design per pass, not a batch

The score CSVs are **87–90 bytes**: a header and a single data row.

```
ID,avg_plddt,ptm,avg_pae
8oep.pdb,97.2927451133728,0.7814092636108398,5.4314351081848145
```

`examples/impress_r/adaptive_rome.py` assumes 6–8 designs per pass. The real
campaign contributes **one record per (pipeline, pass)**. Every one of the 176
CSVs in `p1-p16` has exactly one data row.

That fixes the corpus budget, and it is small:

| | |
|---|---|
| records per target, whole campaign | **11** (uniform across all 16) |
| records for the 16-target group | **176** |
| records per pass, 16 targets | 16 – 32 (median 21) |
| extrapolated to 70 targets | ~770 total, ~70–140 per pass |

So `min_samples` has to be set against a per-pass yield in the tens, not the
hundreds. Combined with the filter below (~32% admission), **`min_samples=24`**
gives roughly one training round per pass at 70 targets. The `min_samples=64` in
the README's snippet would fire about twice in a whole campaign.

### Migration is uniform, and it is what makes 11 records out of 8 passes

Each target's lineage is exactly four nodes deep — `p1`, `p1_sub1`,
`p1_sub1_sub2`, `p1_sub1_sub2_sub3` — for all 70 targets, hitting the
`MAX_SUB_PIPELINES = 3` cap every time. In all 48 handoffs in the `p1-p16`
group, **the child's first pass is the parent's last pass**, not the one after:
a migration re-runs the pass it was triggered on. The three duplicated passes
are what turn 8 passes into 11 records.

```
p11                 p1:0.85 p2:0.87 p3:0.90 p4:0.91 p5:0.93 p6:0.87
p11_sub1                                                    p6:0.84
p11_sub1_sub2                                               p6:0.84
p11_sub1_sub2_sub3                                          p6:0.67 p7:0.76 p8:0.69
```
<sub>pTM by pass. Three migrations fire on pass 6 alone, once the parent's pTM turns down.</sub>

24 of 28 handoffs happen on a pass where pTM fell against the previous pass, so
migration is tracking exactly the degradation ROME-A is meant to fix. Two
consequences for the wiring: the contribution step will see the same
`(target, pass)` more than once and must key on the full `model_id`, and a
lineage that has spent all three migrations has no move left except a better
model.

### The predictor is Boltz, and the prediction path is not what the example assumes

`af_pipeline_outputs_multi` is a legacy name. `sequences_indexed.csv` gives the
real path of every scored structure, and it is Boltz output nested well below
`output_path_af`:

```
{group}/protein_binding/af_pipeline_outputs_multi/{model_id}
    /af/prediction/dimer_models/{target}
    /boltz_results_{target}/predictions/{target}/{target}_model_0.pdb
```

All 280 rows follow this template. Three things fall out of it.

**It is a `dimer_model` — the structure is a complex, not a monomer.** This
settles the open question: `n_prot = 2`, `avg_pae` is an interface signal and
worth ranking on, and the AF3-style weighting should count two protein chains.

**It is keyed by `model_id`, not by pass — so it is overwritten.** `p10_sub1`
runs passes 2 through 6 and writes all five predictions to one
`2pdz_model_0.pdb`. A corpus record storing that path points at a file whose
contents change under it, so by training time it no longer holds the structure
that was scored. **The contribution step has to copy the prediction to a
pass-qualified location before recording it.** Deduplicating on sequence does not
save this — the path *is* the training example. This is the one finding that
forces a code change rather than a retune.

**`ID` is the input PDB name** (`8oep.pdb`), identical every pass, so it cannot
identify a record. Cluster on `ID` for weighting — sub-pipelines chain their
suffixes and would otherwise split one structure across four clusters — but key
records on `(model_id, pass)`.

### `sequences_indexed.csv` is an analysis artifact, not a pipeline output

Worth being explicit, because it is the most convenient-looking file in the
campaign. Its `kmer_cos_dist` and `aa_entropy` columns are post-hoc diversity
metrics and its `confidence` is Boltz's own score — it matches `ptm` in none of
the 64 `p1-p16` rows. It carries **one row per `model_id` with no pass column**,
so it cannot be joined to `af_stats_*` per pass; it is the final state of each
lineage node, 280 rows for 70 targets.

For ROME-A it is still useful for two things: the designed sequences are 76–108
residues (median 91), and **all 280 are distinct**, so the corpus needs no
sequence-level deduplication.

### The filter thresholds were wrong, and pLDDT does not discriminate

All 176 records of the `p1-p16` group:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| `avg_plddt` | 87.96 | 92.30 | **95.73** | 96.99 | 97.83 |
| `ptm` | 0.669 | 0.840 | **0.905** | 0.928 | 0.956 |
| `avg_pae` | 1.99 | 3.14 | **3.81** | 4.82 | 8.01 |

The shipped defaults, `impress_corpus_filter(80, 0.80, 5.0)`, **admit 146 of 176
records — 83%.** Clause by clause: `pLDDT >= 80` admits 100%, `pTM >= 0.80`
admits 89%, `pAE <= 5.0` admits 84%.

This was the mistake, and it was a conceptual one rather than an arithmetic one.
The thresholds were taken from IMPRESS's own keep/drop rule, on the reasoning
that IMPRESS already knows what a good design is. But everything that survives
into the score CSVs has *already* cleared that rule — the filter is being applied
downstream of itself. Reusing it selects nothing and fine-tunes ProteinMPNN on
the campaign's own median output, which is the failure mode ROME-A exists to
avoid.

Retuned defaults, now `impress_corpus_filter(93.0, 0.90, 4.0)`:

| thresholds | admits |
|---|---|
| 80 / 0.80 / 5.0 (old) | 146 (83%) |
| 90 / 0.85 / 4.5 | 105 (60%) |
| **93 / 0.90 / 4.0** | **56 (32%)** |
| 95 / 0.90 / 4.0 | 41 (23%) |
| 97 / 0.90 / 3.0 | 3 (2%) |

pLDDT still barely discriminates — it never drops below 88, so even the raised
93 only removes the bottom quartile — and the pTM/pAE clauses do the real work.
That also breaks the example's sampling: it uses `score_key="pLDDT"` with
`sampling="top_k"`, and ranking on a value spanning 88–98 with a median of 95.7
is close to ranking at random. **Rank by pAE (lower is better) or pTM.**
