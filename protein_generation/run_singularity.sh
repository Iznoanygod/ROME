#!/bin/bash
set -euo pipefail

# Usage: run_fold.sh <input_fasta> [stage_dir] [random_seed]
# Accepts absolute paths for output.
if [ $# -lt 1 ]; then
  echo "Usage: $0 <colabfold_sif> <input_fasta_name> [stage_dir] [cache_dir] [random_seed]" 1>&2
  exit 2
fi

SIFFILE="$1"
INPUTFILE="$2"
STAGEDIR="${3:-${INPUTFILE}.out}"
CACHEDIR="${4:-/work/hdd/bdyk/apark4/foldcache}"
RANDOMSEED="${5:-0}"
#echo "If you can read this, that is good"
#singularity pull ${SIFFILE} docker://ghcr.io/sokrypton/colabfold:1.6.0-cuda12
#echo "If you can read this but not the previous line, this is less good" >&2
singularity run --nv  -B ${CACHEDIR}:/cache -B ${STAGEDIR}:/work   ${SIFFILE}  colabfold_batch /work/${INPUTFILE} /work >&2
#echo "If you can read this, docker is done and this is all very very good"
#echo "If you can read this but not the previous line, docker is done but you dont have output" >&2
#export PATH="/work/nvme/bdyk/apark4/localcolabfold/.pixi/envs/default/bin:${PATH}"

#colabfold_batch \
#  --num-recycle 3 \
#  --templates \
#  --num-models 5 \
#  --model-order 1,2,3,4,5 \
#  --random-seed ${RANDOMSEED} \
#  "${INPUTFILE}" \
#  "${OUTPUTDIR}"