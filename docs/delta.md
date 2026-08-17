# Setting up ROME-A + IMPRESS on Delta

End-to-end: environment, install, a ladder of smoke tests that each prove one
more layer, then running an IMPRESS campaign with ROME-A attached.

**What has actually been verified, and where.** Everything below was run on a
single-node Linux box with Dragon 0.14.1 — not on Delta. The Delta-specific
parts (module names, partitions, account, `sbatch` shape) are marked where they
need checking against your allocation. The software parts — Dragon, asyncflow,
ROME-A, IMPRESS, and the two of them together — were run and are reported with
their real output.

---

## 1. Environment

Delta gives you `/work/nvme` (fast, for code and checkpoints) and `/work/hdd`
(bulk, for structures and caches). The existing scripts in
`protein_generation/` already use `/work/nvme/<project>/<user>/...`, so keep
that layout.

```bash
export PROJ=/work/nvme/bdyk/$USER          # adjust to your project/user
export BULK=/work/hdd/bdyk/$USER
mkdir -p $PROJ $BULK
```

Python 3.11 is what everything below was verified against. Dragon publishes
wheels per CPython version, so the interpreter version is not a free choice —
check what `dragonhpc` has for your Python before committing to one.

```bash
module avail python anaconda          # Delta-specific: see what is offered
python3 -m venv $PROJ/venv-rome
source $PROJ/venv-rome/bin/activate
pip install --upgrade pip
```

A venv is easier than conda here because Dragon, `radical.asyncflow` and
`rhapsody-py` are all plain wheels.

## 2. Install Dragon

```bash
pip install dragonhpc
dragon --version          # Dragon Version 0.14.1
```

Two things to know from the start:

* **Run programs as `dragon -s script.py`**, not `python script.py`. Dragon's
  API imports without the runtime but every object it creates asserts on launch
  parameters that only exist inside a Dragon launch, so plain `python` fails
  with `Launch parameter not initialized: GS_CD`.
* **Run `dragon-cleanup-deprecated` after every Dragon program**, including
  after a crash or a timeout. Leftovers stop the next run from starting.

## 3. Install ROME

```bash
cd $PROJ
git clone <your ROME remote> ROME && cd ROME
git checkout claude/rome-agnostic-implementation-xi5bz4
pip install -e '.[test]'
pip install 'rhapsody-py[dragon]'     # Dragon execution backend for asyncflow
```

Sanity check without any HPC involvement:

```bash
pytest -m fast
```

Expect **175 passed**, plus 5 pre-existing failures in
`tests/integration/test_protein_flow_dummy.py`, which exercise the legacy
`oldrome` protein flow and are unrelated.

## 4. Install IMPRESS

```bash
cd $PROJ
git clone --branch archive/ipdps_pdz_usecase --single-branch \
  https://github.com/radical-collaboration/IMPRESS.git
cd IMPRESS
pip install --no-deps -e .
```

`--no-deps` is deliberate: `pyproject.toml` declares `radical.pilot`, but it is
only needed for the RADICAL execution backend. The examples and the full test
suite run without it.

The archived branch predates an asyncflow rename, so its examples need two
lines changed — see `docs/impress.md` for the detail and the version table.
Applied to a working copy:

```bash
sed -i 's/ConcurrentExecutionBackend/LocalExecutionBackend/g' \
  examples/dummy.py examples/dummy_adaptive.py
python -m pytest -q          # 19 passed
python examples/dummy.py     # 3 pipelines, clean exit
```

---

## 5. Smoke-test ladder

Run these in order on a compute node. Each one proves one more layer, so when
something breaks you know which layer to look at. Every command below was run
and produced the output shown.

```bash
cd $PROJ/ROME
```

**(a) Dragon primitives — is the DDict reachable?**

```bash
dragon -s tests/dragon/test_namespace_dragon.py && dragon-cleanup-deprecated
```
```
ok    single-key round trip
ok    dict records survive pickling
ok    prefix scan is namespace-scoped
ok    drain claims exactly once
...
all DDict/Event checks passed
```

**(b) The whole ROME-A loop — 4 stream replicas, a trainer, one real DDict.**

```bash
dragon -s tests/dragon/test_manager_dragon.py && dragon-cleanup-deprecated
```
```
ok    every request answered exactly once
ok    concurrent writers lose nothing
ok    training fired and published
ok    streams swapped onto the checkpoint
ROME-A works on Dragon
```

**(c) The worked example — watch a model version climb while inference serves.**

```bash
dragon -s examples/agnostic/dummy_loop.py && dragon-cleanup-deprecated
```

**(d) IMPRESS-R — real IMPRESS pipeline, real ROME-A, stubbed executables.**

```bash
dragon -s examples/impress_r/adaptive_rome.py && dragon-cleanup-deprecated
```
```
[PIPELINE-P1] pass 1 | mpnn=proteinmpnn_v_48_020.pt
[PIPELINE-P1] corpus 4 (+4 this pass) | WAITING
[PIPELINE-P1] ROME-A published v1
[PIPELINE-P1] pass 3 | mpnn=.../checkpoints/dummy/v1      <-- campaign swapped
...
[PIPELINE-P1] ROME-A published v7
```

At this point everything but the science executables is proven on your machine.

---

## 6. Moving off one node: the Dragon execution backend

Steps (a)–(d) use `LocalExecutionBackend`, which runs task bodies as threads in
the driver process. For a real allocation you want tasks placed on nodes, which
is `rhapsody`'s Dragon backend — the same one `protein_generation/dragon_protein_run.py`
already uses:

```python
from rhapsody.backends import DragonExecutionBackendV3

backend = DragonExecutionBackendV3({
    "num_nodes": 2,                     # defaults to the whole allocation
    "results_ddict_mem": 4 * 1024**3,   # raise for large returns / many tasks
})
manager = rome.Manager(backend=backend, ...)   # ROME-A builds its own engine
```

`batch_kwargs` are forwarded verbatim to `dragon.workflows.batch.Batch()`.

**Verified on this backend:** the data + training path, which is what IMPRESS-R
uses. A run publishing three checkpoints across processes:

```
cycle 1: corpus 4  | v1 | model=v1
cycle 3: corpus 8  | v2 | model=v2
cycle 5: corpus 12 | v3 | model=v3
```

Two things this turned up that you will hit:

* **A `TrainTask`'s in-process state does not come back.** The task body runs in
  a different process, so anything it records on `self` is invisible to the
  driver — in the run above the trainer object reported `0` rounds while three
  checkpoints were on disk. Only the returned checkpoint path crosses back.
  Write results to the DDict or to disk, not to instance attributes.
* **`DDict.get(key, default)` hangs.** Not raises — hangs. Use `d[key]` in a
  `try/except KeyError`. ROME-A's `Namespace.get` already does; this matters if
  you touch a DDict directly in your own pipeline code.

**Not yet working on this backend: ROME-A's inference/reward streams.** A stream
submitted to the multi-process Dragon backend stays in `STARTING` and its body
never runs. The state layer is not the problem — a task in another process reads
the driver's DDict, writes back visibly, and sets Dragon events, all confirmed —
so it is something in the stream submission path specifically. Streams work on
`LocalExecutionBackend`, and **the IMPRESS-R integration does not use them**
(inference is IMPRESS's own AlphaFold and MPNN tasks), so this does not block
the campaign. It does block using ROME-A to *serve* inference across nodes.

---

## 7. Running a campaign under Slurm

Delta-specific values to confirm first — check with `sinfo` and your allocation:

```bash
sinfo -s                      # partitions, e.g. gpuA100x4
accounts                      # your charge account
```

```bash
#!/bin/bash
#SBATCH --job-name=impress-r
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>     # e.g. gpuA100x4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=impress-r-%j.out

set -euo pipefail
export PROJ=/work/nvme/bdyk/$USER
source $PROJ/venv-rome/bin/activate
cd $PROJ/ROME

# Keep campaign state off the login filesystem.
export ROME_CHECKPOINTS=$PROJ/checkpoints
export IMPRESS_BASE=/work/hdd/bdyk/$USER/campaign
mkdir -p "$ROME_CHECKPOINTS" "$IMPRESS_BASE"

dragon -s examples/impress_r/adaptive_rome.py
rc=$?
dragon-cleanup-deprecated || true       # always, including after a failure
exit $rc
```

Start with `--nodes=1` and the stubbed example to confirm Dragon comes up under
Slurm at all, then scale.

## 8. Swapping in the real science

The example stubs three things. Replacing them is where the remaining work is:

| Stub in the example | Real thing |
|---|---|
| `s1_mpnn` echo | `mpnn_wrapper.py` from IMPRESS's `protien_binding_usecase` |
| `s4_alphafold` echo | `af2_multimer_reduced.sh` |
| `s5_extract` writing a CSV | `plddt_extract_pipeline.py` |
| `DummyTrainer` | `ProteinMPNNTrainer(ProteinMPNNConfig(...))` |

The `adaptive_fn` reads the real CSV schema (`ID, avg_plddt, ptm, avg_pae`) and
the real structure path (`pipeline.output_path_af/{design}.pdb`), but it does
need one change before a production run: **that path is keyed by pipeline, not
by pass, so the next pass overwrites it and `finalize()` deletes it outright.**
The contribution step has to copy the prediction to a pass-qualified location
before recording it, or the corpus points at files that no longer hold the
structure that was scored. See `docs/impress.md`.

Do not reuse `impress_corpus_filter()`'s defaults either — see §9.

For the trainer, read `docs/proteinmpnn_training.md` first: foundry's
ProteinMPNN trains on a **dataframe of structure-file paths, not sequences**,
and the public weights need a one-time conversion before a round can resume from
them. That path needs foundry installed and a GPU, and is the one part of the
chain not yet run.

Two open items to settle before a production run, both noted in
`docs/impress.md` and `docs/proteinmpnn_training.md`:

* IMPRESS's real `run_protein_binding.py` uses `RadicalExecutionBackend`, which
  exists only in asyncflow 0.2.0. On current asyncflow, use the Dragon backend
  above.
* Fine-tuning only on self-generated designs will drift the model, and the
  standard mitigation — mixing in a slice of the original training distribution
  — needs dataframes IPD has not released yet.

## 9. Selecting designs without knowing your thresholds yet

`impress_corpus_filter()`'s defaults admit **83%** of a real campaign. They are
IMPRESS's own keep/drop thresholds, and everything reaching the score CSVs has
already cleared those, so the filter is applied downstream of itself and selects
nothing. Fine-tuning on that corpus trains ProteinMPNN on its own median output.

Replacing them needs a distribution, and confidence scales are predictor
specific — an AlphaFold2-multimer campaign and a Boltz one are not comparable —
so the numbers have to come from the run you are doing. Two ways to get there,
and the first needs nothing up front.

**Rank instead of threshold (recommended for a first run).**

```python
from rome.train.mpnn import percentile_sampler

rome.DataConfig(
    min_samples=24,
    sample_func=percentile_sampler(0.33, on_summary=print),
)
```

"The best third of what this campaign has produced" needs no scale, so it works
on the first round before any distribution exists, and keeps working if you
switch predictors. Ranking is by average rank across pAE (down) and pTM (up), so
the two contribute equally without normalisation and neither's outliers dominate.
`on_summary=print` reports, every round, the corpus size, how many were selected,
and **the cutoffs an equivalent fixed filter would have used** — which is how the
run hands you the calibration data as a byproduct.

Leave `filter_func` off, or keep it only to reject malformed records. Admission
and selection are different jobs; this does the selecting.

**Watch the distribution directly.**

```bash
python scripts/af_stats_watch.py $IMPRESS_BASE --follow
```

Reads every `af_stats_*.csv` written so far and prints the live distribution
plus what each candidate threshold triple would admit. Read-only, safe against a
running job. Once a few passes have landed, pick the row admitting roughly a
third and pass it explicitly:

```python
filter_func=impress_corpus_filter(min_pLDDT=..., min_pTM=..., max_pAE=...)
```

One trap it will show you: setting each of the three clauses at its 33rd
percentile does **not** admit a third. On measured data it admitted 6%, because
the three scores correlate. If you do choose fixed thresholds, verify the joint
admission rate rather than reasoning clause by clause.

Whichever route, `sampling="top_k"` with `score_key="pLDDT"` is worth avoiding:
pLDDT never fell below 88 across 176 measured records, so ranking on it is close
to ranking at random.
