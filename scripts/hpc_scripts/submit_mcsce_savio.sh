#!/bin/bash
#SBATCH --job-name=mcsce_03242026
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3
#SBATCH --array=1-13129%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=4G
#SBATCH --qos=savio_debug
#SBATCH --output=logs/mcsce_03242026.out
#SBATCH --error=logs/mcsce_03242026.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu

# Setup:
#   bash make_pdb_list.sh /path/to/cath20-filtered-foldseek > pdb_list.txt
#   mkdir -p logs
#   sbatch submit_mcsce_savio.sh

module load python
# Activate your conda env with MCSCE installed:
# conda activate mcsce

PDB_LIST="pdb_list.txt"
OUTDIR="ensembles/mcsce"
NCONFS=5
TEMPERATURE=300.0

python run_mcsce.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --temperature "$TEMPERATURE"
