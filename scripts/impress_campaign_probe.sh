#!/bin/bash
# Answer the open IMPRESS-R wiring questions from a campaign directory.
#
#   bash scripts/impress_campaign_probe.sh /path/to/prod > campaign_probe.txt
#
# Produces one bounded text file, typically a few hundred KB, dominated by the
# concatenated score CSVs. It reads; it never writes into the campaign.
#
# What each section settles:
#   1  layout            what a campaign produces, and how much of it
#   2  score CSVs        the real pLDDT/pTM/pAE distribution -> filter thresholds
#   3  AF output tree    whether predictions are kept per pass or overwritten
#   4  chains in a PDB   monomer or complex -> n_prot, filters, weighting alphas
#   5  MPNN seqs         how sequences map to designs -> iter_seqs, num_seqs
#   6  run config        max_passes, num_seqs, thresholds actually used
#   7  timing            pass duration -> how long a training round may take

set -uo pipefail
ROOT="${1:?usage: impress_campaign_probe.sh <campaign-dir>}"
cd "$ROOT" || exit 2
MAX_LINES=200

section() { printf '\n\n===== %s =====\n' "$1"; }

printf 'IMPRESS campaign probe\nroot: %s\ndate: %s\n' "$(pwd -P)" "$(date -u)"

section "1. LAYOUT"
echo "-- top level --"
ls -la | head -40
echo
echo "-- file counts and bytes by extension --"
find . -type f -printf '%s %f\n' 2>/dev/null \
  | awk '{n=split($2,a,"."); ext=(n>1)?a[n]:"<none>"; c[ext]++; b[ext]+=$1}
         END{for(e in c) printf "%-10s %8d files %14d bytes\n", e, c[e], b[e]}' \
  | sort -k4 -rn | head -25
echo
echo "-- directory tree, 3 levels, with child counts --"
find . -maxdepth 3 -type d 2>/dev/null | head -80 | while read -r d; do
  printf '%-60s %s entries\n' "$d" "$(ls -1 "$d" 2>/dev/null | wc -l)"
done

section "2. SCORE CSVs (all of them, concatenated)"
echo "This is the calibration data: it sets the filter thresholds and how many"
echo "records a campaign actually yields."
echo
echo "-- count --"
find . -name 'af_stats_*.csv' 2>/dev/null | wc -l
echo
echo "-- every row, prefixed by its filename --"
find . -name 'af_stats_*.csv' 2>/dev/null | sort | while read -r f; do
  tail -n +2 "$f" 2>/dev/null | sed "s|^|$(basename "$f")\t|"
done

section "3. AF OUTPUT TREE — kept per pass, or overwritten?"
echo "If the same {ID}.pdb path is reused every pass, a corpus record that stores"
echo "it points at a file whose contents changed. Mtimes tell us which."
for d in af_pipeline_outputs_multi/*/; do
  [ -d "$d" ] || continue
  echo
  echo "-- $d --"
  find "$d" -maxdepth 2 -type d 2>/dev/null | head -10
  echo "   ...pdb files with mtimes (first 20):"
  find "$d" -name '*.pdb' -printf '   %TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null \
    | sort | head -20
  echo "   total pdb count: $(find "$d" -name '*.pdb' 2>/dev/null | wc -l)"
  break
done

section "4. CHAINS — monomer or complex?"
echo "Decides n_prot, which filters apply, and the weighting alphas for training."
f=$(find . -path '*af_pipeline_outputs_multi*' -name '*.pdb' 2>/dev/null | head -1)
if [ -n "$f" ]; then
  echo "file: $f"
  echo "-- non-coordinate header lines (first 15) --"
  grep -v '^ATOM' "$f" 2>/dev/null | head -15
  echo "-- distinct chain IDs (col 22 of ATOM records) --"
  awk '/^ATOM/ {print substr($0,22,1)}' "$f" | sort -u | tr '\n' ' '
  echo
  echo "-- residues per chain --"
  awk '/^ATOM/ && substr($0,13,4)==" CA " {print substr($0,22,1)}' "$f" \
    | sort | uniq -c
else
  echo "no PDB found under af_pipeline_outputs_multi"
fi

section "5. MPNN SEQUENCES"
sd=$(find . -type d -name seqs 2>/dev/null | head -1)
if [ -n "$sd" ]; then
  echo "dir: $sd"
  ls -la "$sd" | head -15
  echo "   files in dir: $(ls -1 "$sd" | wc -l)"
  ff=$(find "$sd" -type f | head -1)
  echo
  echo "-- one file in full: $ff --"
  head -"$MAX_LINES" "$ff"
else
  echo "no seqs/ directory found"
fi

section "6. RUN CONFIG"
echo "-- scripts/ --"
ls -la scripts 2>/dev/null | head -20
echo
echo "-- any run/sbatch script, in full --"
for f in $(find . -maxdepth 2 \( -name '*.slurm' -o -name '*.sbatch' -o -name 'run_*.py' -o -name '*.sh' \) 2>/dev/null | head -3); do
  echo "----- $f -----"
  head -80 "$f"
done
echo
echo "-- slurm .out: head and tail --"
for f in $(ls -1 slurm-*.out 2>/dev/null | head -1); do
  echo "----- $f ($(stat -c%s "$f") bytes) -----"
  head -60 "$f"
  echo "   ...[snip]..."
  tail -40 "$f"
done

section "7. TIMING — how long is a pass?"
echo "Sets how long ROME-A has to finish a training round between passes."
find . -name 'af_stats_*.csv' -printf '%TY-%Tm-%Td %TH:%TM:%TS  %f\n' 2>/dev/null \
  | sort | head -60

printf '\n\n===== END =====\n'
