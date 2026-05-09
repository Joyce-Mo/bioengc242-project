#!/bin/bash
#SBATCH --job-name=featurize
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --qos=savio_normal
#SBATCH --output=logs/featurize_%A_%a.out
#SBATCH --error=logs/featurize_%A_%a.err
#SBATCH --array=1-20
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu

# Featurize training-subset PDBs into (C, L, L) .npy stacks for vae.py.
#
# train_pdb_keys.list contains filenames only (no directory prefix), so
# we build a full-path list at runtime. 20 array tasks split ~337k PDBs
# into ~17k each. Already-featurized PDBs are skipped automatically.
#
# CPU-only work -- no GPU needed, uses savio3 partition.
#
# Run BEFORE train_vae_savio.sh or sweep_vae_savio.sh.
#
# To submit:
#   sbatch scripts/hpc_scripts/featurize_array_savio.sh
#
# To chain with training:
#   FEAT_JOB=$(sbatch --parsable scripts/hpc_scripts/featurize_array_savio.sh)
#   sbatch --dependency=afterok:$FEAT_JOB scripts/hpc_scripts/train_vae_savio.sh

set -euo pipefail

module load python

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version

REPO_ROOT="/global/scratch/users/jqmo/rotation3/bioengc242-project"
cd "$REPO_ROOT"

PDB_DIR="/global/scratch/users/jqmo/rotation3/datasets/augmented_ingraham_cath_bugfree"
KEYS_FILE="${PDB_DIR}/train_pdb_keys.list"
FEATURE_DIR="/global/scratch/users/jqmo/rotation3/datasets/ai-cath_vae_features"
N_TASKS=20  # must match --array upper bound

mkdir -p "$FEATURE_DIR" logs

# train_pdb_keys.list has bare filenames; prepend PDB_DIR to get full paths
FULL_PATH_LIST="${FEATURE_DIR}/train_pdb_fullpaths.txt"
if [ ! -f "$FULL_PATH_LIST" ]; then
    sed "s|^|${PDB_DIR}/|" "$KEYS_FILE" > "$FULL_PATH_LIST"
fi

echo "PDB list: $KEYS_FILE ($(wc -l < "$KEYS_FILE") entries)"
echo "Featurize chunk ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"

python vae/featurize_pdb.py \
    --pdb-list "$FULL_PATH_LIST" \
    --outdir "$FEATURE_DIR" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "$N_TASKS"

echo "Done: chunk ${SLURM_ARRAY_TASK_ID}"
