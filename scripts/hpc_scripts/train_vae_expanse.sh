#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=train_vae_expanse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:4
#SBATCH --mem=92G
#SBATCH --time=8:00:00
#SBATCH -o logs/train_vae_expanse_%j.out
#SBATCH -e logs/train_vae_expanse_%j.err
#SBATCH -p gpu
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# End-to-end VAE training on Expanse:
#   1. Featurize augmented AI-CATH PDBs into (7, L, L) .npy stacks
#   2. Train Conv-VAE with best-val checkpointing

set -euo pipefail

nvidia-smi || echo "nvidia-smi not available"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/bioengc242}"
if [ ! -x "$CONDA_ENV_PATH/bin/python" ]; then
    echo "ERROR: $CONDA_ENV_PATH/bin/python not found" >&2
    exit 1
fi
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV=bioengc242

echo "Using python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda, 'is_available', torch.cuda.is_available())" || true

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Repo root: $REPO_ROOT"
cd "$REPO_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/augmented_ingraham_cath_bugfree"
feature_dir="/expanse/lustre/scratch/jmo/temp_project/ai-cath_vae_features"

model_name="vae_cath20_$(date +%Y%m%d)"
output_dir="/expanse/lustre/scratch/jmo/temp_project/${model_name}_${SLURM_JOB_ID}"

mkdir -p "$feature_dir" "$output_dir" logs

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS=100
BATCH_SIZE=64
Z_DIM=64
LR=1e-3
SEED=42

# ── Step 1: Featurize ────────────────────────────────────────────────────────
echo "Step 1/2 — featurizing PDBs"
echo "  source : $PDB_DIR"
echo "  output : $feature_dir"

n_pdbs=$(find "$PDB_DIR" -type f -name "*.pdb" | wc -l)
n_npys=$(find "$feature_dir" -type f -name "*.npy" 2>/dev/null | wc -l)
echo "  found $n_pdbs PDB(s), $n_npys existing .npy(s)"

if [ "$n_npys" -ge "$n_pdbs" ] && [ "$n_pdbs" -gt 0 ]; then
    echo "  all PDBs already featurized — skipping"
else
    python vae/featurize_pdb.py \
        --pdb-dir "$PDB_DIR" \
        --outdir  "$feature_dir"
fi

n_features=$(find "$feature_dir" -type f -name "*.npy" | wc -l)
echo "  $n_features feature stacks ready"
if [ "$n_features" -eq 0 ]; then
    echo "ERROR: no .npy feature files produced" >&2
    exit 1
fi

# ── Step 2: Train VAE ────────────────────────────────────────────────────────
echo "Step 2/2 — training VAE"
echo "  features : $feature_dir"
echo "  output   : $output_dir"
echo "  epochs=$EPOCHS  batch=$BATCH_SIZE  z_dim=$Z_DIM  lr=$LR  seed=$SEED"

python vae/vae.py \
    --feature-dir "$feature_dir" \
    --outdir "$output_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --z-dim "$Z_DIM" \
    --lr "$LR" \
    --seed "$SEED"

echo "Done. Outputs in $output_dir"
ls -lh "$output_dir"
