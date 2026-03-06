#!/usr/bin/env python3
"""
Run Foldseek on ColabFold outputs, using only rank_001 structures.
Usage: python run_foldseek.py --input_dir ./colabfold_output --db afdb50/super --output_dir ./foldseek_results
"""

import argparse
import subprocess
import re 
import shutil
from pathlib import Path


def collect_rank1_structures(input_dir, staging_dir):
    """Copy rank_001 PDB files into a staging directory for batch search."""
    input_path = Path(input_dir)
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r'^(.+)_unrelaxed_rank_001_alphafold2_ptm_model_\d+_seed_000\.pdb$')

    copied = []
    for pdb_file in input_path.glob('*.pdb'):
        if pattern.match(pdb_file.name):
            shutil.copy(pdb_file, staging_path / pdb_file.name)
            copied.append(pdb_file.name)

    return copied



def run_foldseek(staging_dir, db, output_file, tmp_dir, format_output):
    cmd = [
        '/work/hdd/bdyk/apark4/foldseek/bin/foldseek', 'easy-search',
        str(staging_dir),
        db,
        str(output_file),
        str(tmp_dir),
        '--format-output', format_output
    ]


    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Foldseek failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',     required=True)
    parser.add_argument('--db',            required=True)
    parser.add_argument('--output',        required=True, help='Output alignment file')
    parser.add_argument('--staging_dir',   default='./rank1_structures')
    parser.add_argument('--tmp_dir',       default='./foldseek_tmp')
    parser.add_argument('--format_output', default='query,target,lddt,prob')
    args = parser.parse_args()

    print("Collecting rank_001 structures...")
    copied = collect_rank1_structures(args.input_dir, args.staging_dir)
    print(f"Collected {len(copied)} structures into {args.staging_dir}")

    if not copied:
        print("No rank_001 structures found, exiting.")
        return

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    run_foldseek(args.staging_dir, args.db, args.output, args.tmp_dir, args.format_output)
    print(f"Done -> {args.output}")

if __name__ == '__main__':
    main()