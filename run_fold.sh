#!/bin/bash
set -euo pipefail

# Usage: run_fold.sh <input_fasta> [output_dir] [random_seed]
# Accepts absolute paths for both input and output.
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_fasta> [output_dir] [random_seed]" 1>&2
  exit 2
fi

INPUTFILE="$1"
OUTPUTDIR="${2:-${INPUTFILE}.out}"
RANDOMSEED="${3:-0}"
#singularity run --nv  -B /work/hdd/bdyk/apark4/foldcache:/cache -B /work/nvme/bdyk/apark4/ROME:/work   colabfold_1.6.0-cuda12.sif  colabfold_batch /work/stage-chey/sequences.fasta /work/output-chey
export PATH="/work/nvme/bdyk/apark4/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --num-recycle 3 \
  --templates \
  --num-models 5 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  "${INPUTFILE}" \
  "${OUTPUTDIR}"