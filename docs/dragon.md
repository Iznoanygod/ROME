# ROME-A on Dragon

Status: **verified working** on Dragon 0.14.1, single node.

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop
dragon -s examples/agnostic/dummy_loop.py         # the worked example

dragon-cleanup-deprecated                         # after every Dragon run
```

`test_manager_dragon.py` brings up four inference replicas and a trainer over
one real DDict and asserts: 40 requests answered exactly once, work spread over
replicas, 100 records surviving four concurrent writer threads, a training round
firing and publishing, the running streams swapping onto the new checkpoint, and
ROME-A staying inside its key namespace in a DDict shared with the host
workflow. All pass.

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

### Dragon logs during a prefix scan are noisy but harmless

A prefix scan running while another thread deletes the keys it is streaming
produces manager-side tracebacks:

```
dragon.fli.DragonFLIError: Failed to send memory over stream channel
  ... managed_memory.c: dragon_memory_get_size() :: DRAGON_MAP_KEY_NOT_FOUND
```

The manager streams each key's managed memory and the memory is freed under it.
A run produces well over a hundred of these. Measured client-side impact
(2 scanners, 4 deleters): **0 client errors, 0 hangs, every scan returned.** The
client gets a valid, possibly shorter, key list — which is fine, since ROME-A
re-scans on the next poll. Loud, not wrong.

---

## Known limitation: prefix scans do not scale with corpus size

`Namespace` implements prefix scans by listing *every* key in the dictionary and
filtering in Python, because a DDict has no server-side prefix query. Measured:

| keys in the dictionary | one full `keys()` scan |
|---|---|
| 100 | 3.4 ms |
| 1 000 | 18.2 ms |
| 5 000 | 63.2 ms |

Linear, as expected. The problem is the frequency: each stream replica scans on
every poll, and `DataManager.total_count` scans on every trainer poll and on
every `add`. Four replicas at a 20 ms poll interval is ~200 scans/second, so at
5 000 keys the polling alone asks for more DDict work than there is time to do
it in. The corpus and the stream queues share one dictionary, so **corpus growth
slows down inference polling**, which is the coupling to be rid of.

This is why the dummy runs are comfortable and a real campaign would not be.
Nothing here is incorrect — it is a throughput ceiling somewhere around a few
thousand keys.

Two ways out, neither yet chosen:

1. **Give each stream group its own DDict.** Dragon makes this cheap: a DDict
   passes to a task as an object and the receiver attaches itself. The hot-path
   scan then covers only that group's in-flight requests, so it stops growing
   with the corpus. Small change, decouples the two rates, keeps every current
   guarantee.
2. **Make the counters atomic instead of counted.** `DDict.fetch_add` is an
   atomic distributed counter, which would make `total_count` O(1) and remove
   the trainer's poll-loop scan entirely. It comes with a caveat —
   a `fetch_add` value cannot be read back with ordinary `get`/`[]`, only with
   `fetch_add(key, 0)` — and it changes the corpus count from derived-from-truth
   to separately-maintained, which has to stay right across eviction and
   `clear()`.

They are complementary: (1) fixes the stream path, (2) fixes the trainer path.

---

## Operational notes

- **Always run `dragon-cleanup-deprecated` after a Dragon program**, including
  after a crash or a timeout. Leftover processes stop the next run from
  starting.
- **Size the DDict.** Dragon's default `total_mem` is 3 MiB, which a campaign
  corpus will exhaust with an allocation error from inside a manager rather than
  anything ROME-A can explain. `Manager` now requests 1 GiB by default and
  accepts `ddict_kwargs` to override; pass `ddict=` to share the host
  workflow's dictionary instead.
- **ROME-A stays in its namespace.** Every key it writes is prefixed `rome|`,
  asserted by the Dragon test against a shared dictionary.
