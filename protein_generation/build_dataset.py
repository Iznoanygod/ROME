#!/usr/bin/env python3
"""
Build SFT dataset from Foldseek summary and FASTA file.
Usage: python build_dataset.py \
    --foldseek summary.tsv \
    --fasta sequences.fasta \
    --superfamily "Winged helix-like DNA-binding domain superfamily" \
    --output sft_dataset.jsonl \
    --top_fraction 0.5
"""

import argparse
import json


def parse_fasta(fasta_file):
    """Parse FASTA file, return dict of {seq_id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)

    return sequences


def parse_foldseek_summary(foldseek_file):
    """Parse Foldseek summary TSV, return dict of {seq_id: row}."""
    results = {}
    with open(foldseek_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = dict(zip(header, parts))
            results[row['seq_id']] = row
    return results


def build_dataset(foldseek_results, sequences, superfamily, top_fraction):
    # Merge foldseek results with sequences
    merged = []
    missing_seq, missing_fold = 0, 0

    for seq_id, fold_data in foldseek_results.items():
        if seq_id not in sequences:
            missing_seq += 1
            continue
        merged.append({
            'seq_id':      seq_id,
            'sequence':    sequences[seq_id],
            'superfamily': superfamily,
            'reward':      float(fold_data['top_hscore']),
            'top_target':  fold_data['top_target'],
            'top_lddt':    float(fold_data['top_lddt']),
            'avg_lddt':    float(fold_data['avg_lddt']),
            'avg_hscore':  float(fold_data['avg_hscore']),
        })

    for seq_id in sequences:
        if seq_id not in foldseek_results:
            missing_fold += 1

    print(f"Merged:          {len(merged)} sequences")
    if missing_seq:
        print(f"Missing in FASTA:     {missing_seq} seq_ids from Foldseek not found")
    if missing_fold:
        print(f"Missing in Foldseek:  {missing_fold} FASTA sequences have no Foldseek result")

    # Sort by reward descending and take top fraction
    merged.sort(key=lambda x: x['reward'], reverse=True)
    cutoff = int(len(merged) * top_fraction)
    filtered = merged[:cutoff]

    print(f"Top {top_fraction:.0%} cutoff:    {len(filtered)} sequences")
    print(f"Score range:     {filtered[-1]['reward']:.4f} - {filtered[0]['reward']:.4f}")

    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldseek',     required=True, help='Foldseek summary TSV')
    parser.add_argument('--fasta',        required=True, help='FASTA file of generated sequences')
    parser.add_argument('--superfamily',  required=True, help='Superfamily name for prompt construction')
    parser.add_argument('--output',       required=True, help='Output JSONL file')
    parser.add_argument('--top_fraction', type=float, default=0.5)
    args = parser.parse_args()

    print("Parsing FASTA...")
    sequences = parse_fasta(args.fasta)
    print(f"  {len(sequences)} sequences loaded")

    print("Parsing Foldseek summary...")
    foldseek_results = parse_foldseek_summary(args.foldseek)
    print(f"  {len(foldseek_results)} results loaded")

    print("Building dataset...")
    dataset = build_dataset(foldseek_results, sequences, args.superfamily, args.top_fraction)

    with open(args.output, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')

    print(f"\nWritten to {args.output}")
    print(f"\nSample record:\n  {json.dumps(dataset[0], indent=2)}")


if __name__ == '__main__':
    main()