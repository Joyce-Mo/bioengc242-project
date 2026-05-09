#!/bin/bash
#SBATCH --job-name=sweep_vae
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --qos=savio_normal
#SBATCH --output=logs/sweep_vae_%j.out
#SBATCH --error=logs/sweep_vae_%j.err
#SBATCH --array=0-11%2
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu

# Hyperparameter sweep for VAE on Savio.
# Runs 12 configs as a SLURM job array with one GPU per run.
# Each config varies: z_dim, lr, weight_decay, kl_anneal, batchnorm.
#
# After all jobs finish, compare results with:
#   python scripts/evaluation/compare_sweep.py /global/scratch/users/jqmo/rotation3/checkpoints/vae_sweep

set -euo pipefail

nvidia-smi || echo "nvidia-smi not available"

module load python
module load cuda

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

REPO_ROOT="/global/scratch/users/jqmo/rotation3/bioengc242-project"
cd "$REPO_ROOT"

FEATURE_DIR="/global/scratch/users/jqmo/rotation3/datasets/ai-cath_vae_features"
SWEEP_BASE="/global/scratch/users/jqmo/rotation3/checkpoints/vae_sweep"

# Check features exist
n_npys=$(find "$FEATURE_DIR" -type f -name "*.npy" 2>/dev/null | wc -l)
if [ "$n_npys" -eq 0 ]; then
    echo "ERROR: No .npy features found in $FEATURE_DIR"
    echo "Run train_vae_savio.sh first to featurize, or featurize manually:"
    echo "  python vae/featurize_pdb.py --pdb-dir <PDB_DIR> --outdir $FEATURE_DIR"
    exit 1
fi
echo "Found $n_npys feature files in $FEATURE_DIR"

# Sweep grid
# 12 configs: 2 z_dim x 2 lr x 3 regularization combos (dropout fixed at 0.2)
Z_DIMS=(32 64)
LRS=(1e-3 1e-4)
DROPOUT=0.2
# (weight_decay, kl_anneal_epochs, use_batchnorm)
REG_COMBOS=("0.0 0 false" "1e-4 10 false" "0.0 10 true")

# Flatten the grid and pick this task's config
idx=0
for z in "${Z_DIMS[@]}"; do
for lr in "${LRS[@]}"; do
for reg in "${REG_COMBOS[@]}"; do
    if [ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]; then
        read -r wd kl_anneal bn <<< "$reg"
        Z_DIM=$z; LR=$lr; WEIGHT_DECAY=$wd
        KL_ANNEAL=$kl_anneal; USE_BN=$bn
    fi
    idx=$((idx + 1))
done; done; done

RUN_NAME="z${Z_DIM}_lr${LR}_dp${DROPOUT}_wd${WEIGHT_DECAY}_kl${KL_ANNEAL}_bn${USE_BN}"
OUTPUT_DIR="${SWEEP_BASE}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR" logs

echo "=== Config ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} ==="

BN_FLAG=""
if [ "$USE_BN" = "true" ]; then
    BN_FLAG="--use-batchnorm"
fi

python vae/vae.py \
    --feature-dir "$FEATURE_DIR" \
    --outdir "$OUTPUT_DIR" \
    --epochs 100 \
    --batch-size 64 \
    --z-dim "$Z_DIM" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    $BN_FLAG \
    --kl-anneal-epochs "$KL_ANNEAL" \
    --lr-schedule plateau \
    --lr-patience 10 \
    --early-stop-patience 20 \
    --seed 42

echo "Done: $RUN_NAME"
