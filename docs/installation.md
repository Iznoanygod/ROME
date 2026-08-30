# Installation

ROME-A is a Python package. It needs Python **3.9+** for the framework itself,
and Python **3.10–3.12** if you want the Dragon runtime that backs its shared
state on a cluster.

## From a checkout

```bash
git clone https://github.com/iznoanygod/rome.git
cd rome
pip install -e .
```

This pulls in the framework and its trainers:

| Dependency | Why |
| --- | --- |
| `radical.asyncflow` | The workflow engine ROME-A submits every task to. ROME-A schedules nothing itself. |
| `rhapsody-py[radical_pilot]` | Execution backends — RADICAL-Pilot for HPC. |
| `rhapsody-py[dragon]` | The Dragon execution backend (Python 3.10–3.12 only). |
| `torch`, `transformers`, `peft`, `trl`, `datasets` | The built-in LLM/GRPO trainer. |
| `numpy`, `pandas`, `pyarrow` | Dataset assembly, including the ProteinMPNN trainer's parquet shard. |

!!! note "Heavy imports stay out of the driver process"

    `rome.train.llm` imports TRL, transformers and peft *inside* the functions
    that need them, and `rome.train.__init__` imports the GRPO trainer lazily.
    A workflow that never trains an LLM never pays for that stack, even though
    it is installed.

## Test extras

```bash
pip install -e '.[test]'
pytest -m fast          # unit + mocked integration, no GPUs
```

`-m fast` is everything that does not download a model or run real training.
The `slow` marker covers end-to-end tests that do.

## Documentation

To build this site locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000>. `mkdocs build --strict` is what CI runs; it
turns broken internal links and unresolved API cross-references into build
failures.

Note that the docs dependencies are a separate requirements file rather than a
`[docs]` extra: mkdocstrings reads the source **statically** (griffe parses it
rather than importing it), so the API reference builds with none of ROME-A's
runtime dependencies installed — no Dragon, no torch, no GPU. An extra would have
dragged all of them in.

## Dragon

ROME-A keeps its cross-node state in a Dragon `DDict` and its stop/reload
signals in Dragon `Event`s, so `rome.manager` and `rome.stream` import
`dragon.data.ddict` and `dragon.native.event` at module scope. Dragon ships as
part of the `rhapsody-py[dragon]` extra on supported Python versions, but on a
cluster you generally build it against the site's MPI and network stack instead.
[Setting up ROME-A + IMPRESS on Delta](delta.md) walks through that build.

Anything launched under Dragon runs through its launcher rather than plain
`python`:

```bash
dragon examples/agnostic/dummy_loop.py       # single-node
dragon -s examples/agnostic/dummy_loop.py    # multi-node placement
dragon-cleanup-deprecated                    # after every Dragon run
```

The `-s` form is also how ROME-A's Dragon checks are run — they are scripts, not
pytest modules, because the launcher runs a script rather than a test session:

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop, 4 replicas
```

[ROME-A on Dragon](dragon.md) records what running there turned up — notably
that a DDict client handle cannot be shared across threads — and the one known
scaling limit.

## IMPRESS

The IMPRESS-R integration needs IMPRESS installed from the
`archive/ipdps_pdz_usecase` branch, plus a `dauparas/ProteinMPNN` checkout for
the trainer to fine-tune. Neither is a ROME-A dependency — ROME-A is workflow
agnostic, and the ProteinMPNN trainer ships with the example rather than with
the framework. See [Running IMPRESS](impress.md).

## Checking the install

The fastest end-to-end check needs no model, no GPU and no cluster:

```bash
python -c "import rome; print(rome.__version__)"
dragon examples/agnostic/dummy_loop.py
```

If the second one prints a climbing `model v` while the stream keeps serving,
every moving part is working: task placement, the shared dictionary, checkpoint
publication and the hot swap. Go on to the [Quickstart](quickstart.md) for what
that output means.
