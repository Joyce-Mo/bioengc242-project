#!/bin/bash
#SBATCH --job-name=vae_production
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=savio_normal
#SBATCH --output=logs/vae_production_%j.out
#SBATCH --error=logs/vae_production_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu

# Production VAE training on Savio.
# Uses best config from sweep. Update the hyperparameters below after
# running compare_sweep.py on the sweep results.

set -euo pipefail

nvidia-smi || echo "nvidia-smi not available"

module load python
module load cuda

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda, 'is_available', torch.cuda.is_available())" || true

REPO_ROOT="/global/scratch/users/jqmo/rotation3/bioengc242-project"
cd "$REPO_ROOT"

# Paths
FEATURE_DIR="/global/scratch/users/jqmo/rotation3/datasets/ai-cath_vae_features"
OUTPUT_DIR="/global/scratch/users/jqmo/rotation3/checkpoints/vae_production_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" logs

# Verify features exist (featurization must be run separately via
# featurize_array_savio.sh before submitting this job)
n_npys=$(find "$FEATURE_DIR" -type f -name "*.npy" 2>/dev/null | wc -l)
if [ "$n_npys" -eq 0 ]; then
    echo "ERROR: No .npy features in $FEATURE_DIR"
    echo "Run featurize_array_savio.sh first:"
    echo "  FEAT_JOB=\$(sbatch --parsable scripts/hpc_scripts/featurize_array_savio.sh)"
    echo "  sbatch --dependency=afterok:\$FEAT_JOB scripts/hpc_scripts/train_vae_savio.sh"
    exit 1
fi
echo "Found $n_npys feature files in $FEATURE_DIR"

# ── Best hyperparameters from sweep (UPDATE AFTER SWEEP) ─────────────────────
Z_DIM=64
LR=1e-4
BATCH_SIZE=64
DROPOUT=0.2
WEIGHT_DECAY=1e-4
KL_ANNEAL_EPOCHS=10
USE_BATCHNORM=true
LR_SCHEDULE=plateau
LR_PATIENCE=10
EARLY_STOP_PATIENCE=30
EPOCHS=300
SEED=42

BN_FLAG=""
if [ "$USE_BATCHNORM" = "true" ]; then
    BN_FLAG="--use-batchnorm"
fi

echo "=== Production training ==="
echo "  output: $OUTPUT_DIR"
echo "  z_dim=$Z_DIM lr=$LR dropout=$DROPOUT wd=$WEIGHT_DECAY kl_anneal=$KL_ANNEAL_EPOCHS bn=$USE_BATCHNORM"

python vae/vae.py \
    --feature-dir "$FEATURE_DIR" \
    --outdir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --z-dim "$Z_DIM" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    $BN_FLAG \
    --kl-anneal-epochs "$KL_ANNEAL_EPOCHS" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-patience "$LR_PATIENCE" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --seed "$SEED"

echo "Done. Outputs in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
