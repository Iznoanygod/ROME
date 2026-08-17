"""Does a concurrent pop() truncate another client's keys() scan?

    dragon -s keys_pop_race.py

Dragon's KeysOp snapshots the key list, then streams it from a daemon thread
with transfer_ownership=False. A pop() that frees a key's memory before the
thread sends it makes the descriptor dangling. The client's receive loop treats
the resulting early stream termination as EOFError -> done, so the question is
whether a scan silently comes back short.
"""

import multiprocessing as mp
import time

from dragon.data.ddict import DDict

N_KEYS = 400
POPPERS = 3


def popper(serialized, lo, hi, done):
    d = DDict.attach(serialized)
    for i in range(lo, hi):
        try:
            d.pop(f"pop|{i}")
        except (KeyError, TypeError):
            pass
        time.sleep(0.0005)
    done.value += 1
    d.detach()


def main():
    d = DDict(2, 1, 512 * 1024 * 1024)

    # Stable keys nobody touches: any scan must always return all of them.
    for i in range(N_KEYS):
        d[f"keep|{i}"] = i
    # Volatile keys the poppers will delete underneath the scans.
    for i in range(N_KEYS):
        d[f"pop|{i}"] = i

    ser = d.serialize()
    done = mp.Value("i", 0)
    span = N_KEYS // POPPERS + 1
    procs = [
        mp.Process(target=popper, args=(ser, k * span, min((k + 1) * span, N_KEYS), done))
        for k in range(POPPERS)
    ]
    for p in procs:
        p.start()

    scans = short = errors = 0
    worst = N_KEYS
    while done.value < POPPERS:
        try:
            keys = list(d.keys())
        except Exception as ex:
            errors += 1
            print(f"  keys() raised: {type(ex).__name__}: {ex}")
            continue
        scans += 1
        kept = sum(1 for k in keys if isinstance(k, str) and k.startswith("keep|"))
        if kept < N_KEYS:
            short += 1
            worst = min(worst, kept)

    for p in procs:
        p.join()

    print(f"\nscans           : {scans}")
    print(f"keys() raised   : {errors}")
    print(f"scans missing a stable key : {short}")
    print(f"worst stable-key count     : {worst}/{N_KEYS}")

    final = sum(1 for k in d.keys() if isinstance(k, str) and k.startswith("keep|"))
    print(f"quiesced scan   : {final}/{N_KEYS}")

    if short:
        print("\n=> keys() TRUNCATES SILENTLY under concurrent pop()")
    else:
        print("\n=> keys() stayed complete; the thread exception is cosmetic")

    d.destroy()


if __name__ == "__main__":
    mp.set_start_method("dragon", force=True)
    main()
