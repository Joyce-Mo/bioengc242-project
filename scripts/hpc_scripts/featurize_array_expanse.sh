#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=featurize_ai-cath
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs/featurize_ai-cath_04182026_%j.out
#SBATCH -e logs/featurize_ai-cath_04182026_%j.err
#SBATCH -p gpu-shared
#SBATCH --array=1-4
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# Featurize training-subset PDBs into (C, L, L) .npy stacks for vae.py.
#
# train_pdb_keys.list contains bare filenames (no directory prefix), so
# we prepend PDB_DIR at runtime. 4 array tasks split ~337k PDBs into
# ~84k each. Total cost: 4 x 12h = 48 GPU-hours.
#
# Idempotent: already-featurized PDBs (existing .npy) are skipped, so
# if any task times out, just resubmit and it picks up where it left off.
#
# Run BEFORE train_vae_production_expanse.sh or sweep_vae_expanse.sh.

set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version

REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/bioengc242-project"
cd "$REPO_ROOT"

PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/augmented_ingraham_cath_bugfree/mpnn_esmfold"
KEYS_FILE="/expanse/lustre/scratch/jmo/temp_project/augmented_ingraham_cath_bugfree/train_pdb_keys.list"
FEATURE_DIR="/expanse/lustre/scratch/jmo/temp_project/ai-cath_vae_features"
N_TASKS=4  # must match --array upper bound

mkdir -p "$FEATURE_DIR" 

# train_pdb_keys.list has bare filenames, but PDBs live in subdirectories
# under PDB_DIR. Build an index of actual paths, then look up each key.
FULL_PATH_LIST="${FEATURE_DIR}/train_pdb_fullpaths.txt"
echo "Indexing PDB files under ${PDB_DIR} ..."
find "$PDB_DIR" -name '*.pdb' -type f > "${FEATURE_DIR}/_all_pdb_paths.txt"
echo "  Found $(wc -l < "${FEATURE_DIR}/_all_pdb_paths.txt") total PDB files"

# Create a basename -> full-path lookup via awk, then join with the keys file
awk -F/ '{print $NF, $0}' "${FEATURE_DIR}/_all_pdb_paths.txt" | sort -k1,1 \
    > "${FEATURE_DIR}/_pdb_path_index.txt"
sort "$KEYS_FILE" | join -o 2.2 - "${FEATURE_DIR}/_pdb_path_index.txt" \
    > "$FULL_PATH_LIST"

N_FOUND=$(wc -l < "$FULL_PATH_LIST")
echo "  Matched ${N_FOUND} / $(wc -l < "$KEYS_FILE") keys to actual paths"

TOTAL=$(wc -l < "$KEYS_FILE")
N_EXIST=$(find "$FEATURE_DIR" -name '*.npy' 2>/dev/null | wc -l)
echo "PDB list: $KEYS_FILE ($TOTAL entries, $N_EXIST already featurized)"
echo "Featurize chunk ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"

python vae/featurize_pdb.py \
    --pdb-list "$FULL_PATH_LIST" \
    --outdir "$FEATURE_DIR" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "$N_TASKS"

echo "Done: chunk ${SLURM_ARRAY_TASK_ID}"
echo "Total features now: $(find "$FEATURE_DIR" -name '*.npy' | wc -l)"
