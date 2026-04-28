#!/usr/bin/env python3
"""
Parse Foldseek output and summarize per sequence ID.
Usage: python parse_foldseek.py --input results.txt --output summary.txt
"""

import argparse
import re
from collections import defaultdict


def parse_sequence_id(query_name):
    """Extract the numeric sequence ID from the full query filename."""
    m = re.match(r'^(\d+)_unrelaxed_rank_001', query_name)
    return m.group(1) if m else query_name


def parse_foldseek_output(input_file):
    """Parse foldseek results, grouping hits by sequence ID."""
    # {seq_id: [(target, lddt, hscore), ...]} in order of appearance
    hits = defaultdict(list)

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue

            query, target, lddt, hscore = parts[0], parts[1], float(parts[2]), float(parts[3])
            seq_id = parse_sequence_id(query)
            hits[seq_id].append((target, lddt, hscore))

    return hits


def summarize(hits):
    """Compute summary stats per sequence ID."""
    rows = []
    for seq_id, matches in sorted(hits.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        total = len(matches)
        top_target, top_lddt, top_hscore = matches[0]  # first hit is top match
        avg_lddt   = sum(m[1] for m in matches) / total
        avg_hscore = sum(m[2] for m in matches) / total

        rows.append({
            'seq_id':     seq_id,
            'total':      total,
            'top_target': top_target,
            'top_lddt':   top_lddt,
            'top_hscore': top_hscore,
            'avg_lddt':   avg_lddt,
            'avg_hscore': avg_hscore,
        })
    return rows


def write_summary(rows, output_file):
    header = '\t'.join([
        'seq_id', 'total_matches',
        'top_target', 'top_lddt', 'top_hscore',
        'avg_lddt', 'avg_hscore'
    ])

    with open(output_file, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            line = '\t'.join([
                r['seq_id'],
                str(r['total']),
                r['top_target'],
                f"{r['top_lddt']:.4f}",
                f"{r['top_hscore']:.4f}",
                f"{r['avg_lddt']:.4f}",
                f"{r['avg_hscore']:.4f}",
            ])
            f.write(line + '\n')

    print(f"Written {len(rows)} sequence summaries to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Foldseek output file')
    parser.add_argument('--output', required=True, help='Summary output file')
    args = parser.parse_args()

    hits = parse_foldseek_output(args.input)
    rows = summarize(hits)
    write_summary(rows, args.output)

    # Print a quick preview to terminal
    print(f"\n{'seq_id':<10} {'matches':<10} {'top_lddt':<12} {'top_hscore':<12} {'avg_lddt':<12} {'avg_hscore'}")
    print('-' * 66)
    for r in rows:
        print(f"{r['seq_id']:<10} {r['total']:<10} {r['top_lddt']:<12.4f} {r['top_hscore']:<12.4f} {r['avg_lddt']:<12.4f} {r['avg_hscore']:.4f}")


if __name__ == '__main__':
    main()