#!/usr/bin/env python3
"""Bundle a small, representative sample of an IMPRESS campaign directory.

Wiring ROME-A into a campaign needs the campaign's *shape*, not its bulk: what
files exist, how they are named, what the score CSVs contain, and what one of
each artifact actually looks like. That is usually a few MB out of many GB.

    python scripts/collect_campaign_sample.py /path/to/campaign -o sample.tar.gz

What goes in:

* ``MANIFEST.tsv`` — every file in the tree with size and mtime. Text, compresses
  well, and is the part that answers "what does a campaign produce".
* ``SUMMARY.txt`` — file counts and total bytes per extension and per directory
  depth, so the layout is readable at a glance.
* every small text artifact in full — CSV, FASTA, JSON, YAML, logs, ``.out`` —
  since those carry the schemas and the run configuration.
* up to ``--per-ext`` examples of everything else (PDB, CIF, pickles, ...),
  preferring the smallest, so the bundle stays reviewable.

Nothing is anonymized: this copies your files as they are. Look at
``MANIFEST.tsv`` before sharing if the campaign contains anything you would not
put in a repo.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict

#: Extensions copied in full when under --max-text-bytes: schemas and config.
TEXT_EXTS = {
    ".csv", ".tsv", ".fasta", ".fa", ".json", ".yaml", ".yml", ".txt",
    ".log", ".out", ".err", ".cfg", ".ini", ".sh", ".md",
}

#: Never worth sampling — big, opaque, and not informative about wiring.
SKIP_EXTS = {".sif", ".tar", ".gz", ".zip", ".pt", ".pth", ".ckpt", ".a3m", ".db"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            yield path, os.path.relpath(path, root), st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", help="campaign root directory")
    ap.add_argument("-o", "--output", default="campaign_sample.tar.gz")
    ap.add_argument("--per-ext", type=int, default=3,
                    help="examples to copy per non-text extension (default 3)")
    ap.add_argument("--max-text-bytes", type=int, default=2 * 1024 ** 2,
                    help="copy text files in full below this size (default 2MB)")
    ap.add_argument("--max-sample-bytes", type=int, default=5 * 1024 ** 2,
                    help="skip a non-text example larger than this (default 5MB)")
    ap.add_argument("--max-total-bytes", type=int, default=50 * 1024 ** 2,
                    help="stop copying once the bundle reaches this (default 50MB)")
    args = ap.parse_args()

    root = os.path.abspath(args.campaign)
    if not os.path.isdir(root):
        print(f"not a directory: {root}")
        return 2

    entries = list(walk(root))
    if not entries:
        print(f"no files under {root}")
        return 2

    by_ext: dict[str, list] = defaultdict(list)
    for path, rel, st in entries:
        by_ext[os.path.splitext(path)[1].lower() or "<none>"].append((path, rel, st))

    staging = tempfile.mkdtemp(prefix="campaign_sample_")
    payload = os.path.join(staging, "campaign_sample")
    os.makedirs(payload, exist_ok=True)

    # -- manifest: the whole tree, as text ---------------------------------
    with open(os.path.join(payload, "MANIFEST.tsv"), "w") as fd:
        fd.write("bytes\tmtime\tpath\n")
        for _path, rel, st in sorted(entries, key=lambda e: e[1]):
            fd.write(f"{st.st_size}\t{int(st.st_mtime)}\t{rel}\n")

    # -- summary: shape at a glance ----------------------------------------
    with open(os.path.join(payload, "SUMMARY.txt"), "w") as fd:
        total = sum(st.st_size for _p, _r, st in entries)
        fd.write(f"campaign root : {root}\n")
        fd.write(f"files         : {len(entries)}\n")
        fd.write(f"total size    : {human(total)}\n\n")
        fd.write(f"{'ext':<12}{'count':>8}{'bytes':>12}\n")
        for ext, items in sorted(by_ext.items(),
                                 key=lambda kv: -sum(s.st_size for _p, _r, s in kv[1])):
            fd.write(f"{ext:<12}{len(items):>8}"
                     f"{human(sum(s.st_size for _p, _r, s in items)):>12}\n")

        fd.write("\ntop-level entries:\n")
        for name in sorted(os.listdir(root)):
            fd.write(f"  {name}\n")

    # -- copy the informative files ----------------------------------------
    copied = skipped_big = 0
    budget = args.max_total_bytes

    def take(path: str, rel: str) -> bool:
        nonlocal copied, budget
        size = os.path.getsize(path)
        if size > budget:
            return False
        dst = os.path.join(payload, "files", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)
        budget -= size
        copied += 1
        return True

    for ext, items in sorted(by_ext.items()):
        if ext in SKIP_EXTS:
            continue
        if ext in TEXT_EXTS:
            for path, rel, st in sorted(items, key=lambda e: e[2].st_size):
                if st.st_size <= args.max_text_bytes:
                    take(path, rel)
        else:
            # Smallest first: an example is for shape, not for size.
            for path, rel, st in sorted(items, key=lambda e: e[2].st_size)[: args.per_ext]:
                if st.st_size > args.max_sample_bytes:
                    skipped_big += 1
                    continue
                take(path, rel)

    out = os.path.abspath(args.output)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(payload, arcname="campaign_sample")
    shutil.rmtree(staging, ignore_errors=True)

    print(f"scanned  : {len(entries)} files under {root}")
    print(f"copied   : {copied} files"
          + (f" ({skipped_big} examples skipped as too large)" if skipped_big else ""))
    print(f"bundle   : {out}  {human(os.path.getsize(out))}")
    print("\nReview campaign_sample/MANIFEST.tsv before sharing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
