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

## Where ROME-A would attach

Notes from reading the code, for the IMPRESS-R work.

A pipeline subclasses `ImpressBasePipeline` and implements three abstract
methods:

| method | role |
|---|---|
| `register_pipeline_tasks()` | declare tasks via `self.auto_register_task()` |
| `run()` | the per-pipeline sequence |
| `finalize()` | teardown |

`ImpressManager.start(pipeline_setups=[...])` drives them, where each
`PipelineSetup` carries `name`, `type`, `config`, and an optional
`adaptive_fn(pipeline) -> None`. The adaptive function is how a pipeline
reshapes the campaign: `pipeline.submit_child_pipeline_request(config)` queues a
new pipeline, which the manager picks up.

Two things matter for adding ROME-A:

* **`auto_register_task(local_task=True)` leaves the function alone** instead of
  wrapping it as an executable task. That is the seam for a Python-level step —
  such as contributing a scored design to ROME-A's data manager — that should
  not become a shell command.
* **`adaptive_fn` is the natural place to read `manager.get_current_model()`**,
  since it already runs between pipeline stages and already decides what the
  next generation looks like. A pipeline that picks up improved weights there
  needs no change to `run()`.

`ImpressManager` creates its own `WorkflowEngine` internally from the backend
you hand it (`self.flow = await WorkflowEngine.create(backend=...)`). ROME-A
expects to be given an engine, so IMPRESS-R either shares
`manager.flow` after start, or constructs the engine first and passes it to
both. Worth settling before wiring the two together.
