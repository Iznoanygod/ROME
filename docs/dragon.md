# ROME-A on Dragon

Status: **verified working** on Dragon 0.14.1, single node.

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop
dragon -s examples/agnostic/dummy_loop.py         # the worked example

dragon-cleanup-deprecated                         # after every Dragon run
```

`test_manager_dragon.py` brings up four inference replicas and a trainer over
real DDicts — one for the manager, one for the stream group — and asserts: 40
requests answered exactly once, work spread over replicas, 100 records surviving
four concurrent writer threads, a training round firing and publishing, the
running streams swapping onto the new checkpoint, and ROME-A staying inside its
key namespace in a dictionary shared with the host workflow. All pass.

These are scripts rather than pytest modules because the Dragon launcher runs a
script, not a test session. They exit non-zero on failure.

---

## What was broken, and why

### A DDict client handle is not safe to share between threads

This was a hard failure, not a slow path. ROME-A ran the manager's event loop
and its stream workers in one process against one `DDict` object, and the run
died partway through with:

```
dragon.fli.DragonFLIEOT: DragonFLIEOT: End of Transmission
```

raised from an unrelated later operation, which is what makes it awkward to
diagnose — the corruption and the symptom are not the same call.

Reduced to a probe (four threads, 200 operations each):

| handle strategy | result |
|---|---|
| one handle shared by all threads | 3 threads failed with `DragonFLIEOT` |
| `DDict.attach(d.serialize())` per thread | 0 errors |

A handle multiplexes an FLI channel, so concurrent use corrupts it. The fix is
`rome.utils.thread_handle`: every `Namespace` operation goes through a handle
owned by the calling thread, attached once and cached. Plain-mapping backings
(the unit-test path) are returned unchanged, so nothing else moved.

Worth being clear about the scope, since Dragon *does* handle transport for
you: passing a DDict object into a task on another node attaches the receiving
**process** automatically, and that has always worked. Threads inside one
process are the gap.

### The test suite broke as soon as Dragon was installed

`conftest.py` stubbed Dragon only when the import failed. But Dragon's API
imports fine without a runtime and then asserts on launch parameters the moment
you construct anything:

```
AssertionError: Launch parameter not initialized: GS_CD
```

So installing Dragon made plain `pytest` fail, which is exactly backwards. The
check is now whether the runtime is *up* (`DRAGON_GS_CD` in the environment),
not whether the package is importable.

---

## What was confirmed rather than fixed

### `pop` is an exactly-once claim

ROME-A's claim protocol is "whoever pops the key owns the item". Four threads
racing over 120 keys: **120 claimed, 120 unique, 0 duplicates.** The losers get
`DDictKeyError`, which subclasses `KeyError`, so ROME-A's existing handling was
already correct.

`Namespace.pop` now delegates to the backing's `pop` instead of doing
read-then-delete: one round trip instead of two, and the contention is resolved
manager-side rather than in a window between two calls. Note that
`DDict.pop(key, default)` raises regardless of the default, unlike
`dict.pop` — so the single-argument form plus `except KeyError` is the only
spelling that behaves identically on both backings.

---

## Prefix-scan cost, and why each stream group owns a dictionary

`Namespace` implements prefix scans by listing *every* key in the dictionary and
filtering in Python, because a DDict has no server-side prefix query. Measured:

| keys in the dictionary | one full `keys()` scan |
|---|---|
| 100 | 3.4 ms |
| 1 000 | 18.2 ms |
| 5 000 | 63.2 ms |

Linear, as expected. The problem was never the scan itself but what shared it:
with the corpus and the stream queues in one dictionary, every replica's poll
paid for the corpus, so **corpus growth slowed inference polling** — two rates
that have nothing to do with each other.

So each stream group now gets its own DDict, allocated by the stream manager and
released by `Stream.close()`. A group's dictionary holds only requests in flight
and results not yet collected, which does not accumulate. Measured with a real
Manager, timing a request round trip as the corpus grows:

| corpus records | keys in the group's dict | median request RTT |
|---|---|---|
| 0 | 2 | 54.8 ms |
| 500 | 2 | 58.3 ms |
| 2 000 | 2 | 58.0 ms |

Flat, and the group's dictionary stays at two keys throughout. The manager's own
dictionary is still where the corpus and the published checkpoint live; streams
read it only to notice a new checkpoint.

Supplying `StreamConfig.ddict` shares a dictionary you already own instead, in
which case ROME-A will not destroy it.

### What is still O(corpus)

`DataManager.total_count` counts by scanning, and `ready_to_train()` calls it on
every training-manager poll. That is far cooler than a stream poll — the trainer
polls on the order of seconds, not milliseconds — but it does grow with the
corpus, and it is why populating a corpus record-by-record while reading the
count back is quadratic.

`DDict.fetch_add` is the natural fix: an atomic distributed counter would make
this O(1) and remove the scan entirely. It is not done here because it has a
sharp edge worth deciding on deliberately — a `fetch_add` value cannot be read
back with ordinary `get`/`[]`, only with `fetch_add(key, 0)`, so that key stops
behaving like the rest of the dictionary. It also moves the corpus count from
derived-from-truth to separately-maintained, which then has to stay right across
eviction and `clear()`. The scan is slower but self-correcting, which for the
value that decides when training fires is the safer default until someone
chooses otherwise.

### Dragon's manager threads still log during scans

A scan running while another thread deletes the keys it is streaming produces
manager-side tracebacks:

```
dragon.fli.DragonFLIError: Failed to send memory over stream channel
  ... managed_memory.c: dragon_memory_get_size() :: DRAGON_MAP_KEY_NOT_FOUND
```

The manager streams each key's managed memory and the memory is freed under it.
Measured client-side impact (2 scanners, 4 deleters): **0 client errors, 0
hangs, every scan returned** a valid, possibly shorter, key list — which is fine,
since ROME-A re-scans on the next poll. Loud, not wrong.

## Operational notes

- **Always run `dragon-cleanup-deprecated` after a Dragon program**, including
  after a crash or a timeout. Leftover processes stop the next run from
  starting.
- **Size the DDicts.** Dragon's default `total_mem` is 3 MiB, which a campaign
  corpus will exhaust with an allocation error from inside a manager rather than
  anything ROME-A can explain. `Manager` now requests 1 GiB by default and
  accepts `ddict_kwargs` to override; pass `ddict=` to share the host
  workflow's dictionary instead.
- **ROME-A stays in its namespace.** Every key it writes is prefixed `rome|`,
  asserted by the Dragon test against a shared dictionary.
