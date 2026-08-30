# Shared state

Every ROME-A component reads and writes the same distributed dictionary. That is
what lets a task on one node add training data that a training task on another
node picks up, and what lets a stream replica in a third process see that a new
checkpoint has been published.

The dictionary is a Dragon `DDict`. `rome.utils` is the small amount of glue that
makes it pleasant to work with — and the place where the rules that make it
*correct* live.

## The one rule

!!! danger "Never read-modify-write a shared container"

    ```python
    d = ddict["corpus"]      # read
    d[uid] = record          # modify
    ddict["corpus"] = d      # write
    ```

    This loses records the moment two nodes do it at once. It is the obvious way
    to store a corpus in a key-value store, and it is wrong.

So ROME-A gives **every record, request and result its own key**, and rebuilds
collections by scanning a key prefix. A single-key write in a DDict is atomic, so
nothing is ever clobbered, no matter how many producers there are.

```text
rome|record|8oep3f21…     -> {"uid": "8oep…", "sequence": "MKT…", "score": 91.2}
rome|record|a1b2c3d4…     -> {...}
rome|meta|consumed        -> 24
rome|meta|total           -> 28
rome|model_path           -> "/scratch/ckpt/mpnn/v3/v_48_020.pt"
rome|model_version        -> 3
```

The corpus is also **monotonic**: adding is the only thing the host workflow does
to it. Filtering, deduplication and sampling all happen on the way *out*. That is
not just tidiness — it is what makes the concurrent case trivial, because there is
no shared mutation to order.

## `Namespace`

[`Namespace`][rome.utils.Namespace] is a prefixed *view* over a mapping. It holds
no state of its own, so several views over the same dictionary compose freely:

```python
ns       = Namespace(ddict, "rome|")
records  = ns.namespace("record")        # keys under rome|record|
meta     = ns.namespace("meta")          # keys under rome|meta|
requests = ns.namespace("req", 2)        # keys under rome|req|2|
```

It is a `MutableMapping`, so it behaves like a dict — plus prefix-aware
operations:

| Method | What it does |
| --- | --- |
| `keys(prefix="")` | Namespace-relative keys, optionally filtered further. |
| `items()` / `values()` | Pairs under a prefix, **skipping keys that vanish mid-scan**. |
| `drain(prefix, limit)` | Pop and return up to `limit` pairs. The claim primitive. |
| `increment(key, n)` | Bump an integer counter. Only safe when one component owns it — ROME-A keeps to that. |
| `snapshot()` | A plain `dict` copy, for logging and debugging. |

Because any mapping works underneath, every ROME-A component is unit-testable
against a plain `dict` with no Dragon installed. That is why the test suite runs
`pytest -m fast` on a laptop.

## `pop` is an exactly-once claim

ROME-A's claim protocol is one line: **whoever pops a key owns that item.**

```python
batch = self._requests.drain(limit=batch_size)
```

Two stream replicas draining the same queue therefore never process a request
twice. Correctness rests on the underlying pop being atomic, which is why
`Namespace.pop` delegates to the backing dictionary rather than doing
read-then-delete: a Dragon DDict resolves the contention manager-side.

This was measured, not assumed — **four threads racing over 120 keys claim 120
with zero duplicates.**

`Namespace.pop` is deliberately single-argument at the DDict layer:
`dict.pop(k, default)` returns the default, while `DDict.pop(k, default)` raises
`DDictKeyError` regardless. Asking for a default would behave differently on the
two backings, so ROME-A catches `KeyError` (which `DDictKeyError` subclasses)
instead, and behaves identically on both.

## A DDict handle is not thread-safe

!!! warning "This is the bug that cost the most to find"

    A Dragon DDict handle multiplexes an FLI channel and is **not** safe to share
    between threads. Concurrent use corrupts the channel and surfaces as
    `dragon.fli.DragonFLIEOT`, usually raised from an unrelated, later operation —
    so the traceback points nowhere near the cause.

    It is not hypothetical. ROME-A runs the manager's event loop and its
    `asyncio.to_thread` stream workers in one process against one dictionary,
    which reproduces it within a few hundred operations.

Dragon's own remedy is that each consumer attaches its own handle, so that is
what [`thread_handle`][rome.utils.thread_handle] does: one handle per thread,
cached by the dictionary's serialized form. Attaching is cheap and the thread
pool is bounded, so the cache stays small.

Every `Namespace` operation goes through it:

```python
@property
def _target(self):
    return thread_handle(self._ddict, self._serialized)
```

The dictionary is serialized **once**, in `__init__`, rather than per access —
`serialize()` is itself a client operation, and calling it from many threads would
be the very race this is meant to avoid. Child namespaces inherit the serialized
form rather than recomputing it, because child views are created on hot paths.

When the backing is a plain mapping (`serialized is None`), it is returned
unchanged. That is the unit-test and single-process case.

Full account: [ROME-A on Dragon](../dragon.md).

## Why each stream group owns a dictionary

A DDict has **no server-side prefix query**. A scan lists every key in the
dictionary and filters client-side, so the cost of a scan is proportional to the
whole dictionary — not to the part you asked for.

If stream queues shared the manager's dictionary, a replica polling for work
every 100 ms would scan the entire corpus each time. That couples two rates with
nothing to do with each other: **inference polling frequency** and **campaign
corpus size**. A long campaign would slow its own inference down.

So each stream group gets its own dictionary:

```mermaid
flowchart TB
    M[("manager DDict<br/><small>1 GiB</small><br/>corpus, meta,<br/>model_path, model_version")]
    G1[("'generate' DDict<br/><small>256 MiB</small><br/>req / out / status")]
    G2[("'score' DDict<br/><small>256 MiB</small><br/>req / out / status")]

    M -. "read-only:<br/>published checkpoint" .-> G1
    M -. .-> G2
```

A group's dictionary holds only requests in flight and results not yet collected.
It does not accumulate, so a replica's poll stays cheap for the whole campaign.
The manager's dictionary is still read from a stream — read-only, for the
published checkpoint — but that is a single-key get, not a scan.

Group dictionaries default to 256 MiB against the manager's 1 GiB, for the same
reason. Supply your own with `StreamConfig(ddict=...)` and ROME-A will not
destroy it on teardown.

## What is still O(corpus)

Two things scan the corpus, and both are on cold paths:

* **`get_dataset()`** — once per training round, which is the point.
* **`_is_duplicate()`** — once per `add()` when `dedup_key` is set, past the
  per-process fast-path set.

Deduplication is the one that could bite a very large campaign. If it does, the
answer is a dedicated identity namespace (one key per identity, an O(1) get)
rather than a corpus scan.

## `keys()` truncates silently under concurrent pops

!!! danger "The reason `max_records` carries a warning"

    A Dragon `DDict.keys()` scan **silently returns a truncated list** when
    another client pops concurrently. It does not raise, and it does not report
    a partial result — one measured scan returned **39 of 400 keys** and reported
    success.

    Eviction is the only thing that pops from the *corpus* dictionary. With
    `max_records` unset there are no evictions, so corpus scans are exact and
    every training shard is complete. That is why
    [`DataConfig.max_records`][rome.data.DataConfig] should stay `None` on Dragon
    unless you have measured that you need it.

Stream group dictionaries *are* popped constantly, by design — but nothing there
needs an exact scan. A replica claiming 7 of 8 available requests just claims the
eighth on its next poll, and `items()` skips keys that vanish mid-scan rather
than raising.

## Key reference

| Key | Written by | Read by |
| --- | --- | --- |
| `rome\|record\|<uid>` | `DataManager.add` | `DataManager.get_records` |
| `rome\|meta\|consumed` | `DataManager.mark_consumed` | `unconsumed_count` |
| `rome\|model_path` | `Trainer._publish` | streams, `get_current_model()` |
| `rome\|model_version` | `Trainer._publish` | streams (reload trigger), record stamping |
| `rome\|last_trained_at`, `rome\|last_train_samples` | `Trainer._publish` | reporting |
| `req\|<i>\|<rid>` *(group dict)* | `Stream.submit` | replica `i`'s claim |
| `out\|<rid>` *(group dict)* | `StreamTask.emit` | `get_outputs` |
| `status\|<i>` *(group dict)* | replica `i` | `get_status`, cross-node |

The `rome|` prefix is what makes sharing the host workflow's dictionary safe:
[`ROME_NS`][rome.manager.ROME_NS] isolates every key ROME-A writes.
