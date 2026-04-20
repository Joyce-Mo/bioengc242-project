#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=featurize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o logs/featurize_04192026_%j.out
#SBATCH -e logs/featurize_04192026_%j.err
#SBATCH -p gpu-shared
#SBATCH --array=1-100%4
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# Parallel featurization of PDBs into (C, L, L) .npy stacks for vae.py.
#
# Splits the full PDB list into 100 chunks and processes one chunk per
# SLURM array task. Each chunk is ~3,400 PDBs, which takes roughly
# 2-4 hours depending on protein size.
#
# Run BEFORE train_vae_production_expanse.sh or sweep_vae_expanse.sh.
#
# To submit:
#   sbatch scripts/hpc_scripts/featurize_array_expanse.sh
#
# To submit training after featurization completes:
#   FEAT_JOB=$(sbatch --parsable scripts/hpc_scripts/featurize_array_expanse.sh)
#   sbatch --dependency=afterok:$FEAT_JOB scripts/hpc_scripts/train_vae_production_expanse.sh

set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version

REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/bioengc242-project"
cd "$REPO_ROOT"

PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/augmented_ingraham_cath_bugfree"
FEATURE_DIR="/expanse/lustre/scratch/jmo/temp_project/ai-cath_vae_features"
N_TASKS=100  # must match --array upper bound

mkdir -p "$FEATURE_DIR" logs

echo "Featurize chunk ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"

python vae/featurize_pdb.py \
    --pdb-dir "$PDB_DIR" \
    --outdir "$FEATURE_DIR" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "$N_TASKS"

echo "Done: chunk ${SLURM_ARRAY_TASK_ID}"
