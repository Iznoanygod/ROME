"""How many repeated keys() scans does it take to see a complete key set?

    dragon -s keys_union.py

Truncation only ever omits keys, never invents them, so unioning N scans is
monotonically safer. This measures how large N has to be.
"""

import multiprocessing as mp
import time

from dragon.data.ddict import DDict

N_KEYS = 400
POPPERS = 3
TRIALS = 12


def popper(serialized, lo, hi, done):
    d = DDict.attach(serialized)
    for i in range(lo, hi):
        try:
            d.pop(f"pop|{i}")
        except (KeyError, TypeError):
            pass
        time.sleep(0.002)
    done.value += 1
    d.detach()


def union_scan(d, passes):
    seen = set()
    for _ in range(passes):
        try:
            seen.update(k for k in d.keys() if isinstance(k, str))
        except Exception:
            pass
    return seen


def main():
    d = DDict(2, 1, 512 * 1024 * 1024)
    for i in range(N_KEYS):
        d[f"keep|{i}"] = i

    results = {}
    for passes in (1, 2, 3, 4):
        for i in range(N_KEYS):
            d[f"pop|{i}"] = i

        done = mp.Value("i", 0)
        span = N_KEYS // POPPERS + 1
        procs = [
            mp.Process(target=popper,
                       args=(d.serialize(), k * span, min((k + 1) * span, N_KEYS), done))
            for k in range(POPPERS)
        ]
        for p in procs:
            p.start()

        complete = trials = 0
        worst = N_KEYS
        while done.value < POPPERS and trials < TRIALS:
            seen = union_scan(d, passes)
            kept = sum(1 for k in seen if k.startswith("keep|"))
            trials += 1
            complete += (kept == N_KEYS)
            worst = min(worst, kept)
        for p in procs:
            p.join()

        results[passes] = (complete, trials, worst)
        print(f"passes={passes}: complete {complete}/{trials} scans, "
              f"worst {worst}/{N_KEYS} stable keys")

    print(f"\n{'passes':<8}{'complete scans':<18}{'worst view':<14}")
    for passes, (complete, trials, worst) in results.items():
        print(f"{passes:<8}{f'{complete}/{trials}':<18}{f'{worst}/{N_KEYS}':<14}")

    d.destroy()


if __name__ == "__main__":
    mp.set_start_method("dragon", force=True)
    main()
