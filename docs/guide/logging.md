# Logging

ROME schedules training out of sight of the host workflow, so a campaign
operator needs to *see* what it is doing: when a design arrives, when a round is
submitted, when a new model is published.

`rome._logging` wires a single stdout handler on the `rome` logger namespace
whose output mirrors IMPRESS's `impress.utils.logger.ImpressLogger` — the same
`HH:MM:SS.mmm [LEVEL] [COMPONENT]` shape and the same per-level and
per-component colours — so ROME's lines sit alongside IMPRESS's
`[PIPELINE-P1]` lines in one run.

```text
12:34:56.789 [INFO] [ROME-DATA]    received design 8oep (pLDDT=95.0) — corpus 8 (8 unconsumed)
12:34:57.001 [INFO] [ROME-TRAINER] submitting training round 1 (8 designs, trainer mpnn) -> v1
12:34:58.512 [INFO] [ROME-MODEL]   published v1 (8 designs) -> .../v_48_020.pt
12:34:58.530 [INFO] [ROME-STREAM]  infer[0] reloaded weights -> v1 (v_48_020.pt)
```

It matches IMPRESS's *format* without importing IMPRESS, so ROME keeps working
in workflows that have nothing to do with it — the LLM/GRPO trainer, the dummy
loop.

## Components

The tag is the segment after `rome.`, so each manager announces itself:

| Tag | Source | Colour |
| --- | --- | --- |
| `[ROME-DATA]` | `rome.data` — records accepted and rejected | bright yellow (matches IMPRESS `data`) |
| `[ROME-TRAINER]` | `rome.trainer` — rounds submitted, waits, failures | bright magenta |
| `[ROME-MODEL]` | checkpoint publication | bright green (matches IMPRESS `checkpoint`) |
| `[ROME-STREAM]` | `rome.stream` — groups started, replicas reloading | bright cyan |
| `[ROME-MANAGER]` | `rome.manager` — start/stop | bright white |

Publication logs under `[ROME-MODEL]` rather than `[ROME-TRAINER]` because a
round publishing a checkpoint *creates a model* — it is the event the campaign
operator is actually watching for.

## Environment

| Variable | Effect |
| --- | --- |
| `ROME_LOG_LEVEL` | `INFO` (default), `DEBUG` for per-record detail — including why each rejected record was rejected — or `WARNING` to quiet the lifecycle lines. |
| `ROME_LOG_COLOR` | `0` disables ANSI colour. Also disabled when `NO_COLOR` is set. Default on, like IMPRESS, since Dragon captures a non-tty stdout. |

```bash
ROME_LOG_LEVEL=DEBUG dragon examples/agnostic/dummy_loop.py
```

`DEBUG` adds one line per rejected record with the reason — `score=72.1 < 80.0`,
`no pLDDT`, `filter_func`, `duplicate` — which is the fastest way to find out why
a corpus is not filling up.

## Keeping your own configuration

The handler is attached **once, lazily, and only if the `rome` logger has none**.
An application that configures its own logging keeps control:

```python
import logging

logging.getLogger("rome").handlers = [my_handler]   # before importing/using rome
```

ROME's logger sets `propagate = False` when it attaches its own handler, so it
owns its output and never double-prints into a root handler.

## Turning it down

Two dials, depending on what you want quiet:

```python
logging.getLogger("rome.data").setLevel(logging.WARNING)   # one component
```

```bash
ROME_LOG_LEVEL=WARNING                                     # all of ROME
```

Lifecycle lines are all `INFO`; genuine problems — a round whose result the
backend has not delivered, a failed round — are `WARNING` and `ERROR`, so
`WARNING` still surfaces anything you need to act on.
