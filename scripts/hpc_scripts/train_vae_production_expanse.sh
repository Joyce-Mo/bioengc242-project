#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=vae_production
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/vae_production_%j.out
#SBATCH -e logs/vae_production_%j.err
#SBATCH -p gpu
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# Production VAE training (Checkpoint 5).
# Uses best config from sweep. Update the hyperparameters below after
# running compare_sweep.py on the sweep results.
#
# To update after sweep:
#   1. Run: python scripts/evaluation/compare_sweep.py /expanse/.../vae_sweep
#   2. Copy the best run's hparams into the variables below

set -euo pipefail

nvidia-smi || echo "nvidia-smi not available"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda, 'is_available', torch.cuda.is_available())" || true

REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/bioengc242-project"
cd "$REPO_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/augmented_ingraham_cath_bugfree"
FEATURE_DIR="/expanse/lustre/scratch/jmo/temp_project/ai-cath_vae_features"
OUTPUT_DIR="/expanse/lustre/scratch/jmo/temp_project/vae_production_${SLURM_JOB_ID}"

mkdir -p "$FEATURE_DIR" "$OUTPUT_DIR" logs

# ── Featurize (skip if already done) ─────────────────────────────────────────
n_pdbs=$(find "$PDB_DIR" -type f -name "*.pdb" 2>/dev/null | wc -l)
n_npys=$(find "$FEATURE_DIR" -type f -name "*.npy" 2>/dev/null | wc -l)

if [ "$n_npys" -ge "$n_pdbs" ] && [ "$n_pdbs" -gt 0 ]; then
    echo "Features already exist ($n_npys .npy files) — skipping featurization"
else
    echo "Featurizing $n_pdbs PDBs..."
    python vae/featurize_pdb.py --pdb-dir "$PDB_DIR" --outdir "$FEATURE_DIR"
fi

# ── Best hyperparameters from sweep (UPDATE AFTER SWEEP) ─────────────────────
# These are reasonable defaults; replace with actual best config from sweep.
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
