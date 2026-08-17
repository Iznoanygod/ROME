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

From the `IMPRESS data` Drive folder (`prod_in_70` inputs, `prod/p1-p16` results
of a 70-pipeline PDZ run). These correct several assumptions the example was
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
campaign contributes **one record per (pipeline, pass)**. Across 70 pipelines and
8 passes that is a few hundred records for a whole campaign, so `min_samples`
has to be set against that budget, not against a per-pass batch.

### The design ID is the input structure, and its path is reused

`ID` is the input PDB name (`8oep.pdb`), the same value every pass. Two
consequences for the wiring:

* **`output_path_af/{design}.pdb` is overwritten each pass.** A corpus record
  that stores that path points at a file whose contents change under it, so by
  training time it no longer holds the structure that was scored. The
  contribution step has to copy the prediction to a pass-qualified location
  before recording it. Deduplicating on sequence does not save this — the path
  is the training example.
* **Clustering by `pipeline.name` splits one structure across many keys.**
  Sub-pipelines chain their suffixes (`p9_sub1_sub2_sub3`, up to the
  `MAX_SUB_PIPELINES = 3` cap), so the same `8oep` appears under several
  pipeline names and would be weighted as several clusters. Cluster on the `ID`
  column instead.

### pLDDT does not discriminate; pTM and pAE do

Two passes of the same sub-pipeline:

| pass | pLDDT | pTM | pAE | `impress_corpus_filter(80, 0.80, 5.0)` |
|---|---|---|---|---|
| 3 | 97.26 | 0.819 | 4.91 | accept |
| 8 | 97.29 | 0.781 | 5.43 | **reject** (pTM < 0.80, pAE > 5.0) |

pLDDT sits at ~97 throughout — nowhere near the 80 threshold, which is therefore
inert. The binding constraints are pTM and pAE, and they sit *right on* the
default thresholds, so the accept rate is genuinely sensitive to them.

That also breaks the example's sampling: it uses `score_key="pLDDT"` with
`sampling="top_k"`, and ranking by a value that is ~97 for everything is close
to ranking at random. Rank by pAE (lower is better) or pTM instead.

The trajectory is also the degradation IMPRESS is built to detect — pTM falling
and pAE rising from pass 3 to pass 8 — which is consistent with this pipeline
having exhausted its three sub-pipeline migrations.
