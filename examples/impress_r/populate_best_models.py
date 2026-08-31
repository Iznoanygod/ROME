#!/usr/bin/env python3
"""Populate best_models/ and best_ptm/ from dimer_models/ — the step IMPRESS's
AlphaFold task does in its ``post_exec``, which only runs on RadicalExecutionBackend.

IMPRESS selects the best AlphaFold model by copying, per target:

    dimer_models/{target}/*ranked_0*.pdb      -> best_models/{target}.pdb
    dimer_models/{target}/*ranking_debug*.json -> best_ptm/{target}.json

Those copies live in the ``post_exec`` of the AF task, a RADICAL-Pilot feature.
On LocalExecutionBackend or the Dragon backend ``post_exec`` is ignored, so
``best_models``/``best_ptm`` stay empty and ``plddt_extract_pipeline.py`` — which
iterates ``best_models`` — writes a header-only ``af_stats`` CSV.

This does those copies after the fact, so you can re-run the extractor without
re-running AlphaFold. It reads ``dimer_models``; it only writes into
``best_models`` and ``best_ptm``.

    python examples/impress_r/populate_best_models.py --path <base> --out p1
    # or point straight at the prediction dir:
    python examples/impress_r/populate_best_models.py --prediction <base>/af_pipeline_outputs_multi/p1/af/prediction

If a target has no ``ranked_0`` PDB, it prints what the target directory *does*
contain — which is how you tell an AlphaFold run (``ranked_0.pdb``,
``ranking_debug.json``) from a Boltz/other run (``.cif``, ``*_model_0.*``), a
different failure with a different fix.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import shutil


def _find(directory: str, pattern: str) -> str | None:
    for name in sorted(os.listdir(directory)):
        if fnmatch.fnmatch(name, pattern):
            return os.path.join(directory, name)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", help="the af/prediction directory directly")
    ap.add_argument("--path", help="campaign base path (with --out)")
    ap.add_argument("--out", help="pipeline name, e.g. p1 (with --path)")
    args = ap.parse_args()

    if args.prediction:
        prediction = args.prediction
    elif args.path and args.out:
        prediction = os.path.join(args.path, "af_pipeline_outputs_multi", args.out,
                                  "af", "prediction")
    else:
        ap.error("give --prediction, or --path and --out")

    dimer = os.path.join(prediction, "dimer_models")
    best_models = os.path.join(prediction, "best_models")
    best_ptm = os.path.join(prediction, "best_ptm")
    if not os.path.isdir(dimer):
        print(f"no dimer_models under {prediction}")
        return 2
    os.makedirs(best_models, exist_ok=True)
    os.makedirs(best_ptm, exist_ok=True)

    targets = [d for d in sorted(os.listdir(dimer))
               if os.path.isdir(os.path.join(dimer, d))]
    if not targets:
        # dimer_models exists but holds no per-target subdirectories — the AF
        # output is laid out differently than IMPRESS expects.
        print(f"dimer_models has no target subdirectories; it contains:")
        for name in sorted(os.listdir(dimer))[:20]:
            print(f"    {name}")
        return 1

    copied = missing = 0
    for target in targets:
        tdir = os.path.join(dimer, target)
        ranked = _find(tdir, "*ranked_0*.pdb")
        ranking = _find(tdir, "*ranking_debug*.json")
        if ranked and ranking:
            shutil.copyfile(ranked, os.path.join(best_models, f"{target}.pdb"))
            shutil.copyfile(ranking, os.path.join(best_ptm, f"{target}.json"))
            copied += 1
            print(f"ok    {target}: {os.path.basename(ranked)} -> best_models/{target}.pdb")
        else:
            missing += 1
            print(f"MISS  {target}: no ranked_0 pdb / ranking_debug json. Contains:")
            for name in sorted(os.listdir(tdir))[:15]:
                print(f"          {name}")

    print(f"\ncopied {copied} target(s); {missing} without AlphaFold ranked output")
    if copied:
        print("Now re-run plddt_extract_pipeline.py to fill the af_stats CSV.")
    if missing and not copied:
        print("Nothing looked like AlphaFold output — if these are .cif/Boltz "
              "results, the pipeline's best-model copy and the extractor both "
              "need adjusting for that predictor.")
    return 0 if copied else 1


if __name__ == "__main__":
    raise SystemExit(main())
