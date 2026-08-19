#!/bin/bash

# AlphaFold2 multimer (reduced_dbs) on NCSA Delta.
# Called by the IMPRESS protein-binding pipeline as:
#   af2_multimer_reduced.sh <input_fasta_dir> <input_fasta_name> <output_dir>

set -e
set -x
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=".75"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

INPUT_FASTA_FILE_DIR=$1
INPUT_FASTA_FILE_NAME=$2
OUTPUT_DATA_DIR=$3

# AlphaFold2 container and databases (Delta)
AF_CONTAINER=/scratch/rhaas/SUP-5301/alphafold.sif
AF_DB=/scratch/rhaas/SUP-5301/database

# writable /etc inside the container (required by the Delta container setup)
AF_ETC=${AF_ETC:-/scratch/bdyk/apark4/alphafold/etc}
mkdir -p $AF_ETC $OUTPUT_DATA_DIR

#echo $1 $2 $3 >> $NVME/res.txt

singularity run --nv \
  --bind $INPUT_FASTA_FILE_DIR:/fasta \
  --bind $OUTPUT_DATA_DIR:/dimer_models \
  --bind $AF_ETC:/etc \
  --bind $AF_DB:/database \
  --pwd /app/alphafold \
  $AF_CONTAINER \
  --data_dir=/database \
  --uniref90_database_path=/database/uniref90/uniref90.fasta \
  --mgnify_database_path=/database/mgnify/mgy_clusters_2022_05.fa \
  --template_mmcif_dir=/database/pdb_mmcif/mmcif_files/ \
  --obsolete_pdbs_path=/database/pdb_mmcif/obsolete.dat \
  --fasta_paths=/fasta/$INPUT_FASTA_FILE_NAME \
  --output_dir=/dimer_models \
  --model_preset=multimer \
  --db_preset=reduced_dbs \
  --small_bfd_database_path=/database/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --uniprot_database_path=/database/uniprot/uniprot.fasta \
  --pdb_seqres_database_path=/database/pdb_seqres/pdb_seqres.txt \
  --max_template_date=2020-12-01 \
  --use_gpu_relax=False \
  --num_multimer_predictions_per_model=1 \
#  --run_relax=False
#echo "AF2 finished but stuff not doing things i guess" >> $NVME/res.txt