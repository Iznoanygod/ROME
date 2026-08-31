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

The only other requirement is `radical-asyncflow`, which ROME already needs.

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
   reproducing the archive, but ROME needs 0.5.0 — its streams are submitted
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

## IMPRESS-R: where ROME attaches

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
[PIPELINE-P1] ROME published v1
[PIPELINE-P1] pass 3 | mpnn=/tmp/impress_r_.../checkpoints/dummy/v1     <-- swapped
[PIPELINE-P1] corpus 12 (+6 this pass) | WAITING
...
[PIPELINE-P1] pass 10 | mpnn=/tmp/impress_r_.../checkpoints/dummy/v6
[PIPELINE-P1] ROME published v7
```

The campaign runs its passes with baseline weights, ROME publishes v1 after
pass 2, and **pass 3 onward runs MPNN with the campaign's own checkpoint**. The
per-pass acceptance count climbs (+4, +2, +6, +8) as the model improves, which
is the loop closing.

### `adaptive_fn` is the seam

In the protein-binding use case, `adaptive_decision(pipeline)` runs after the
pLDDT-extraction task of every pass, reads
`af_stats_{name}_pass_{n}.csv`, and decides which designs regressed. That is the
one point in the campaign that both *has* fresh scored designs and is *between*
passes, so both halves of ROME go there:

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

`run()` never mentions ROME. The degradation logic that spawns child
pipelines is untouched.

Two details that fall out of the real code:

* **`pipeline.output_path_af/{design}.pdb` is the training example.** It is the
  AlphaFold prediction of the designed sequence — coordinates and sequence in
  one file, which is exactly what ProteinMPNN trains on, with no threading step.
  The real `adaptive_decision` already copies these files when migrating a
  backbone, so the path is known-good. See `docs/proteinmpnn_training.md`.
* **The score columns are `ID, avg_plddt, ptm, avg_pae`**, and IMPRESS's own
  criterion reads the last one: a *rising* `avg_pae` means degraded. Feed
  `avg_plddt` to ROME as the corpus score and let
  `impress_corpus_filter()` apply all three thresholds.

### Which workflow engine

Both work, and `rome.Manager` now supports either:

```python
rome.Manager()                        # builds its own engine at start(),
                                      # shuts it down at stop()
rome.Manager(impress_manager.flow)    # shares the campaign's engine
```

Giving ROME its own is the default in the example: its training rounds are
then scheduled independently of the campaign's tasks, which is what "ROME
handles its own task management" means in practice. Sharing puts rounds and
campaign tasks in one engine against one allocation — better when the two must
compete for the same fixed resources.

One wrinkle if you share: `ImpressManager` creates its engine *inside*
`start()`, and `start()` does not return until the whole campaign is done. So
`impress_manager.flow` does not exist when you would want to construct the
ROME manager. Either build the engine yourself first and hand it to both, or
let ROME build its own. Both paths are covered by
`tests/integration/test_impress_r.py`.

### The training round runs as a command, in its own process

ROME submits a ProteinMPNN round the way IMPRESS submits `mpnn_wrapper.py`: as
an **executable task**, not a Python function pickled into a worker.
`ProteinMPNNTrainer.as_command` stages the round's structures, writes a
self-contained job spec, and returns

```
python examples/impress_r/mpnn_train_wrapper.py --job <output_dir>/train_job.json
```

which the manager runs on the backend with `{"gpus_per_rank": 1}`. Two
consequences matter for a campaign:

* the fine-tune is a **separate process on its own GPU** — nothing about it lives
  in the campaign driver, and the process exits when the round ends, so its VRAM
  is released rather than held for the whole run;
* `examples/impress_r/mpnn_train_wrapper.py` is dragon-free and runnable on its own, so a
  failing round can be reproduced by hand from the `train_job.json` the manager
  left behind.

Point `ProteinMPNNConfig.train_script` at a copy of the wrapper staged elsewhere
on the cluster if the bundled path is not reachable from the compute node, and
supply any environment `pre_exec` through `TrainerConfig.task_description`.

### Other seams worth knowing

* **`auto_register_task(local_task=True)` leaves the function alone** instead of
  wrapping it as an executable task. That is how a Python-level step — anything
  that should not become a shell command — lives inside a pipeline.
* **`PipelineSetup(kwargs={...})`** passes arbitrary configuration through to
  the pipeline constructor, which is how the example threads `base_path`
  through.

### Seeing what ROME is doing — logging that matches IMPRESS

ROME schedules its training out of the campaign's sight, so it logs its own
lifecycle to stdout in the same shape as IMPRESS's `ImpressLogger` — the lines
sit right alongside the `[PIPELINE-P1]` ones in a single run:

```
14:28:32.007 [INFO] [ROME-DATA]    received design 8oep1234 (score=95) — corpus 8 (4 unconsumed)
14:28:33.114 [INFO] [ROME-TRAINER] submitting training round 1 (8 designs, ProteinMPNNTrainer) -> v1
14:28:58.512 [INFO] [ROME-MODEL]   published v1 (8 designs) -> .../v_48_020.pt
14:28:58.520 [INFO] [ROME-STREAM]  generate[0] reloaded weights -> v1 (v_48_020.pt)
```

The events the components emit at `INFO`:

* `[ROME-DATA]` — every design received into the corpus (and, at `DEBUG`, every
  one rejected by a filter or as a duplicate);
* `[ROME-TRAINER]` — each training round submitted, and any round that failed
  (`ERROR`);
* `[ROME-MODEL]` — each new checkpoint published (the "creates a new model"
  event), green like IMPRESS's `checkpoint`;
* `[ROME-STREAM]` — a stream group starting, and a replica reloading weights;
* `[ROME-MANAGER]` — start and stop.

Three environment variables tune it, no code change:

* `ROME_LOG_LEVEL` — `INFO` (default), `DEBUG` for per-record accept/reject
  detail, `WARNING` to quiet the lifecycle lines.
* `ROME_LOG_COLOR=0` — drop the ANSI colour (also dropped when `NO_COLOR` is
  set). On by default, since Dragon captures a non-tty stdout.

It matches IMPRESS's *format* without importing IMPRESS, so the same logging
works in a workflow that has nothing to do with IMPRESS. If an application
configures the `rome` logger itself, ROME leaves it alone.

### `post_exec` only runs on RadicalExecutionBackend — the empty `af_stats` trap

The protein-binding pipeline selects AlphaFold's best model in the AF task's
**`post_exec`**:

```python
"post_exec": [
    f"cp {models_path}/*ranked_0*.pdb {best_model_pdb}",       # -> best_models/
    f"cp {models_path}/*ranking_debug*.json {best_ptm_json}",  # -> best_ptm/
    f"cp {models_path}/*ranked_0*.pdb {mpnn_pdb}",
]
```

`pre_exec`/`post_exec`/`output_staging` are **RADICAL-Pilot** task-description
features. The upstream `run_protein_binding.py` runs on `RadicalExecutionBackend`,
where they execute. On `LocalExecutionBackend` or the Dragon backend — the only
options on current asyncflow, since `RadicalExecutionBackend` is 0.2.0 only —
**`post_exec` is silently ignored**. So AlphaFold fills `dimer_models/` but
nothing copies the ranked model into `best_models/`/`best_ptm/`, and
`plddt_extract_pipeline.py` — whose outer loop is `for files in
os.listdir(best_models)` — writes a **header-only `af_stats` CSV**. Three
symptoms, one cause: `dimer_models` populated, `best_models`/`best_ptm` empty,
`af_stats` empty.

Two remedies:

* **Already-run campaign:** `examples/impress_r/populate_best_models.py` does the copies
  from an existing `dimer_models/`, so you can re-run the extractor without
  re-running AlphaFold. It also prints a target's contents when there is no
  `ranked_0` PDB, which distinguishes an AlphaFold run from a Boltz/other one.
* **New runs:** `examples/impress_r/protein_binding_rome.py` uses
  `ProteinBindingPipelineR`, which folds the copies into the AF task's own shell
  command with `&&` so they run on any backend, right after AlphaFold.


---


## What the campaign source says (archive branch, AlphaFold2)

Read directly out of `examples/protien_binding_usecase/` and
`src/impress/pipelines/protein_binding.py` at `baf42a8`. These hold regardless
of which predictor a given campaign ran, because they are IMPRESS's own logic.

### The design is a dimer: a designed PDZ domain plus a fixed 10-mer peptide

`s3` writes the AlphaFold input FASTA as two chains:

```python
design_seq = self.iter_seqs[base_name][self.seq_rank][0]
pep_seq = "EGYQDYEPEA"
f.write(f">pdz\n{design_seq}\n>pep\n{pep_seq}\n")
```

and `plddt_extract_pipeline.py` computes `avg_pae` over exactly the cross terms
between the last 10 residues and the rest:

```python
target_range = range(length-10, length)
if operator.xor(row_index in target_range, col_index in target_range):
    running_sum += values3[row_index][col_index]
```

So: **`n_prot == 2`**, `avg_pae` is a genuine *interface* pAE, and the peptide is
a **constant 10-mer, identical for every design and every pipeline**. Only chain
A carries designed sequence.

For the ProteinMPNN trainer this settles the open question in
`docs/proteinmpnn_training.md` §5 — but note the asymmetry: the structure has two
chains while only one is a design target, so foundry's stock `n_prot == 1` filter
rejects every example and the weighting alphas want setting for a two-chain
complex where one chain is fixed context.

### One scored design per pass, whatever `num_seqs` is

`num_seqs` defaults to 10, so MPNN emits 10 sequences per pass — but `s3` selects
a single one by rank (`self.iter_seqs[base_name][self.seq_rank][0]`) and only
that one is folded and scored. `fasta_list_2` is the contents of `{name}_in`,
which the campaign inputs show is one PDB per pipeline.

**The corpus therefore grows by one record per (pipeline, pass)** — the other
nine MPNN sequences are never scored and cannot enter it. That is the binding
constraint on ROME here, and it is not a tunable: raising `num_seqs` produces
more sequences but not more *labelled* ones.

### The prediction path is keyed by pipeline, not by pass

```python
self.output_path_af = os.path.join(
    self.output_path, "af/prediction/best_models")     # {base}/af_pipeline_outputs_multi/{name}/...
```

and the extractor takes only `--out` (the pipeline name) for the path, using
`--iter` solely to name the output CSV. So every pass of a pipeline writes its
prediction to the same `best_models/{design}.pdb`.

**A corpus record storing that path goes stale**, in two distinct ways: the next
pass overwrites the file, and `finalize()` calls
`os.unlink(f"{self.output_path_af}/{a}.pdb")` outright. So the contribution step
has to copy the prediction to a pass-qualified location before recording it.
Deduplicating on sequence does not help — the path *is* the training example.

This is the one finding that forces a code change, and it is confirmed on the
branch ROME targets, so it is safe to act on.

---

## What the campaign *data* says — and why it cannot be acted on yet

A 70-target PDZ campaign was measured: all 176 `af_stats_*.csv` from the
`p1-p16` group, plus `sequences_indexed.csv` covering all 70 targets.

**That campaign was not run on this branch.** Its structure paths run through
`.../dimer_models/{target}/boltz_results_{target}/predictions/{target}/{target}_model_0.pdb`
— it used **Boltz**. The `archive/ipdps_pdz_usecase` tree contains *zero*
occurrences of "boltz" and ships `af2_multimer_reduced.sh`, and its extractor
reads AlphaFold result pickles (`result_{rank}.pkl`) via `pd.read_pickle`. The
two are different predictors behind an identical CSV schema and an identical
`af_pipeline_outputs_multi/{pipeline}/af/prediction/` prefix, which is exactly
why the mismatch is easy to miss.

**Confirmed by the data and consistent with the source** — safe to rely on:

* `ID, avg_plddt, ptm, avg_pae`, one data row per file, in all 176 CSVs.
* Lineage is uniform: every target reaches `p{n}_sub1_sub2_sub3`, hitting the
  `MAX_SUB_PIPELINES = 3` cap. In all 48 handoffs the child's first pass is the
  **parent's last pass**, not the one after — a migration re-runs the pass it
  fired on. That is what turns 8 passes into 11 records per target.
* `ID` is the input PDB name, identical every pass, so it cannot identify a
  record. Cluster on it for weighting; key records on `(pipeline_name, pass)`.
* All 280 designed sequences are distinct (76–108 residues, median 91), so the
  corpus needs no sequence-level deduplication.

**Boltz-specific — do not calibrate against these:**

* *Score distributions.* pLDDT median 95.7 (min 88.0), pTM 0.905, pAE 3.81.
  These are Boltz confidences; AF2-multimer's are not on the same scale.
* *Timing.* The 16-target campaign ran in 1h49m with passes starting ~9 min
  apart. Boltz is far faster than AF2-multimer with MSAs, so an AF2 campaign's
  pass budget will be substantially longer.
* *`max_passes`.* The data shows 8; the archive branch defaults to 4.

### The filter defaults are wrong, and deliberately left wrong

`impress_corpus_filter(80, 0.80, 5.0)` admits **146 of 176 records — 83%**.
Clause by clause: `pLDDT >= 80` admits 100%, `pTM >= 0.80` 89%, `pAE <= 5.0` 84%.

The error is conceptual, not arithmetic, and it is predictor-independent: the
thresholds were taken from IMPRESS's own keep/drop rule, but everything that
reaches the score CSVs has already cleared that rule. **The filter is applied
downstream of itself**, so it selects nothing and ROME fine-tunes ProteinMPNN
on the campaign's own median output — the failure it exists to prevent.

The fix needs a percentile, and a percentile needs a distribution from the right
predictor. Retuning on the Boltz numbers (which would give ~93/0.90/4.0 for the
top third) would just relocate the mistake onto a scale AF2 does not share, so
the defaults stay as they are with a warning in the docstring until an AF2
campaign's `af_stats_*.csv` are available.

Two things are worth fixing regardless of predictor, because they do not depend
on the scale:

* **Rank by pAE or pTM, not pLDDT.** The example uses `score_key="pLDDT"` with
  `sampling="top_k"`. pLDDT never dropped below 88 in 176 records, so ranking on
  it is close to ranking at random. pAE has the widest spread (2.0–8.0) and is
  the interface metric IMPRESS's own degradation criterion reads.
* **Select a *fraction*, not a threshold.** Given one scored design per pipeline
  per pass, a filter that expresses "the best third of what this campaign has
  produced" transfers across predictors in a way that a fixed pTM cutoff does
  not.

### What would settle it

`af_stats_*.csv` from a campaign run on `archive/ipdps_pdz_usecase` with
`af2_multimer_reduced.sh` — even one pipeline group over a few passes. The
`examples/impress_r/impress_campaign_probe.sh` sections 2 and 6 collect exactly that.

## Campaign helper scripts

`examples/impress_r/` ships three operational tools alongside the integration
code. **None of them is part of the framework**: nothing in `rome/` imports them,
they carry no ROME API, and `pyproject.toml` packages only `rome*`, so
`pip install rome` does not ship them. They live with the example because that is
what they are specific to — an IMPRESS campaign — and they are things *you* run
from a checkout, by hand, around a run, closer to the `dragon -s` checks than to
library code.

| Script | When you need it |
| --- | --- |
| `populate_best_models.py` | **Whenever you run IMPRESS off RadicalExecutionBackend.** |
| `af_stats_watch.py` | While a campaign runs, to see what a filter would admit. |
| `impress_campaign_probe.sh` | Once, against a finished campaign, to answer wiring questions. |

### `populate_best_models.py` — the one you will actually need

IMPRESS selects the best AlphaFold model by copying, per target:

```text
dimer_models/{target}/*ranked_0*.pdb       -> best_models/{target}.pdb
dimer_models/{target}/*ranking_debug*.json -> best_ptm/{target}.json
```

Those copies live in the AlphaFold task's `post_exec` — a RADICAL-Pilot feature.
On `LocalExecutionBackend` or the Dragon backend **`post_exec` is silently
ignored**, so `best_models/` and `best_ptm/` stay empty,
`plddt_extract_pipeline.py` iterates an empty directory and writes a header-only
`af_stats` CSV, and ROME's corpus receives nothing at all. The failure is quiet:
the campaign appears to run, and training simply never fires.

This does the copies after the fact, so the extractor can be re-run without
re-running AlphaFold. For new runs, `examples/impress_r/protein_binding_rome.py`
avoids the problem entirely — its `ProteinBindingPipelineR` folds the copies into
the AF task's own shell command, where every backend runs them.

### `af_stats_watch.py` — reading a live campaign's distribution

Reads whatever `af_stats_*.csv` exist so far and reports the pLDDT/pTM/pAE
distribution plus a table of candidate thresholds and the fraction each admits.
Read-only, safe against a directory a live job is writing.

```bash
python examples/impress_r/af_stats_watch.py /path/to/campaign            # once
python examples/impress_r/af_stats_watch.py /path/to/campaign --follow   # every 60s
```

It exists because confidence thresholds are predictor-specific. If you use
[`percentile_sampler(0.33)`](guide/data.md#percentile-sampling-when-you-dont-know-your-thresholds)
you do not need it — that does the same calibration continuously, inside the run,
and needs no numbers from you.

### `impress_campaign_probe.sh` — one-shot campaign forensics

```bash
bash examples/impress_r/impress_campaign_probe.sh /path/to/prod > campaign_probe.txt
```

Produces one bounded text file covering the campaign's layout, its score CSV
distributions, whether predictions are kept per pass or overwritten, monomer vs
complex chain structure, how sequences map to designs, and pass timing. It reads;
it never writes. Sections 2 and 6 are what would settle the open threshold
question above.
