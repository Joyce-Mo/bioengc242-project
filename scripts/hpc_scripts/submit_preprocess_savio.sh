#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=savio2
#SBATCH --array=1-13129%500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=logs/preprocess_%a.out
#SBATCH --error=logs/preprocess_%a.err
#
# Preprocess PDB dataset into .pt graph files on Savio
#
# Setup:
#   bash make_pdb_list.sh /path/to/cath20-filtered-foldseek > pdb_list.txt
#   mkdir -p logs
#   sbatch submit_preprocess_savio.sh

module load python
# conda activate your_env

PDB_LIST="pdb_list.txt"
OUT_DIR="processed_graphs"

python preprocess_dataset.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --out_dir "$OUT_DIR"
