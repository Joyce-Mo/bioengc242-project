#!/bin/bash
#SBATCH --job-name=backrub
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3
#SBATCH --array=1-13129%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --qos=savio_debug
#SBATCH --output=logs/backrub_03252026.out
#SBATCH --error=logs/backrub_03252026.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu


# Setup:
#   bash make_pdb_list.sh /path/to/cath20-filtered-foldseek > pdb_list.txt
#   mkdir -p logs
#   sbatch submit_backrub_savio.sh

module load python
# Load Rosetta (adjust path if needed)
export ROSETTA_BIN="/global/scratch/users/jqmo/rotation3/rosetta/main/source/bin/backrub.default.linuxgccrelease"

PDB_LIST="pdb_list.txt"
OUTDIR="/global/scratch/users/jqmo/rotation3/datasets/ai-cath_backrub_ensembles"
NCONFS=5
NSTEPS=10000 # recommended on documentation to run 10000
KT=0.6 # recommended lower temperatures? default of 0.6, using 0.6 like smith & kortemme papern 

python scripts/run_backrub.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --nsteps "$NSTEPS" \
    --kT "$KT" \
    --mode cli \
    --trajectory_gz \
    --trajectory_stride 100 \
    --rosetta_bin "$ROSETTA_BIN"
