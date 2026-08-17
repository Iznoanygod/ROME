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

### `keys()` truncates silently while another client pops

Upgraded from "loud, not wrong" after measuring it properly. It is wrong.

A scan running while another client deletes keys produces manager-side
tracebacks:

```
dragon.fli.DragonFLIError: Failed to send memory over stream channel
  ... managed_memory.c: dragon_memory_get_size() :: DRAGON_MAP_KEY_NOT_FOUND
```

**The cause.** `KeysOp.perform()` snapshots the key list, then hands it to a
*daemon thread* to stream out, and the send takes no ownership
(`send_mem(key, transfer_ownership=False)`). Any `pop` that frees one of those
keys before the thread reaches it leaves a dangling descriptor. The send thread
dies, the stream terminates early, and the client's receive loop treats early
termination as `EOFError -> done` — a normal end of stream.

**So the caller gets a short list and no error.** Measured with 400 stable keys
that nobody touched, scanned while three processes popped 400 other keys
(`tests/dragon/test_keys_race_dragon.py`):

```
scans           : 21
keys() raised   : 0
scans missing a stable key : 11
worst stable-key count     : 39/400
quiesced scan   : 400/400
```

A scan returned **39 of 400** keys and reported success. Nothing was lost from
the dictionary — a quiesced scan sees all 400 — only the *view* is truncated.

**Retrying does not fix it.** Truncation only ever omits, so unioning repeated
scans is monotonically safer, but it does not converge:

| passes | complete scans | worst view |
|---|---|---|
| 1 | 3/12 | 78/400 |
| 2 | 6/12 | 175/400 |
| 3 | 6/12 | 127/400 |
| 4 | 2/4 | 229/400 |

**Prefix namespacing does not protect you either.** `Namespace.keys(prefix)`
calls `self._target.keys()` for the whole dictionary and filters in Python, so
scanning one namespace still streams every key in the dictionary, including the
ones another namespace is popping. Only a *separate DDict* isolates a scan from
a pop.

#### What this means for ROME-A

| dictionary | pops during a run | scans | exposure |
|---|---|---|---|
| manager (corpus, checkpoint) | none, with `max_records=None` (the default) | `get_records`, `total_count`, `unconsumed_count` | **none** |
| stream group | constant — claim-by-pop | pending counts, result drains | truncation, self-healing |

The corpus is the part that matters, because `get_records()` builds the training
shard and nothing re-checks its length — a truncated scan would silently train on
a fraction of the corpus. It is safe **only because nothing pops from that
dictionary**: eviction is the sole popper and `max_records` defaults to `None`.

> **Setting `max_records` turns on eviction, which pops from the corpus
> dictionary, which exposes every corpus scan to silent truncation.** Leave it
> unset unless you have measured that you need it.

Stream groups pop constantly, so their scans do truncate — but every stream
protocol re-polls, and a claim is verified by the `pop` itself, so a short scan
costs a poll interval rather than a result. That is why
`test_manager_dragon.py` passes all seven checks while printing these
tracebacks.

The real fix is structural: give the request queue and the result queue separate
dictionaries, so a scan of one never streams keys the other is popping. Not yet
done.

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

### A busy service task wedges dispatch — on a small node only

**Retracted as a general finding.** `tests/dragon/test_busy_service_blocks_dragon.py`
shows a service task that continuously scans a DDict and pops what it finds
stopping every later task from being dispatched:

```
q1 (no service)      RAN
q2 (idle service)    RAN
q3 (busy service)    BLOCKED      <- on a 4-CPU sandbox
```

On an NCSA Delta GPU node the same script prints `q3 ... RAN`. So this is a
starvation artifact of a small allocation — a service polling at 20 Hz on a box
with about two usable task slots — and **not** the reason a training round
stalls while streams are running. Keep the script as an allocation smoke test;
do not treat it as a Dragon bug.

**What is still unexplained.** On Delta, with 6/6 concurrent service tasks
available and this script passing, `test_manager_dragon.py` still reports no
training round completed. Ruled out, each with its own probe:

* *Capacity.* Reproduces with free slots.
* *Picklability.* The round's body pickles to an identical 2006 bytes with and
  without a stream running.
* *Event-loop starvation.* Worst measured lag 4 ms, a corpus scan 0.16 s, and an
  ordinary task submitted from the same driver still runs.
* *Poll frequency.* 0.05 s to 5 s changes nothing.
* *Dataset size.* 100 records train fine with no stream; 8 do not train with one.
* *The trainer itself.* Alone on this backend it publishes `v1` in ~2 seconds.

The training assertion in `test_manager_dragon.py` now reports trainer status
and the corpus counts, so the next run on a real allocation should say which
link broke rather than only that no round completed.
