#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=sweep_vae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/sweep_vae_%A_%a.out
#SBATCH -e logs/sweep_vae_%A_%a.err
#SBATCH -p gpu
#SBATCH --array=0-23
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# Hyperparameter sweep for VAE (Checkpoint 4).
# Runs 24 configs as a SLURM job array — one GPU per run.
# Each config varies: z_dim, lr, dropout, weight_decay, kl_anneal, batchnorm.
#
# After all jobs finish, compare results with:
#   python scripts/evaluation/compare_sweep.py /expanse/lustre/scratch/jmo/temp_project/vae_sweep

set -euo pipefail

nvidia-smi || echo "nvidia-smi not available"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

FEATURE_DIR="/expanse/lustre/scratch/jmo/temp_project/ai-cath_vae_features"
SWEEP_BASE="/expanse/lustre/scratch/jmo/temp_project/vae_sweep"

# ── Sweep grid ────────────────────────────────────────────────────────────────
# 24 configs: 2 z_dim x 2 lr x 2 dropout x 3 regularization combos
Z_DIMS=(32 64)
LRS=(1e-3 1e-4)
DROPOUTS=(0.0 0.2)
# (weight_decay, kl_anneal_epochs, use_batchnorm)
REG_COMBOS=("0.0 0 false" "1e-4 10 false" "0.0 10 true")

# Flatten the grid and pick this task's config
idx=0
for z in "${Z_DIMS[@]}"; do
for lr in "${LRS[@]}"; do
for dp in "${DROPOUTS[@]}"; do
for reg in "${REG_COMBOS[@]}"; do
    if [ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]; then
        read -r wd kl_anneal bn <<< "$reg"
        Z_DIM=$z; LR=$lr; DROPOUT=$dp; WEIGHT_DECAY=$wd
        KL_ANNEAL=$kl_anneal; USE_BN=$bn
    fi
    idx=$((idx + 1))
done; done; done; done

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
