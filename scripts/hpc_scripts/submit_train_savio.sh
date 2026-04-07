#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00 ## this will be increased for real training run, set shorter for debugging
#SBATCH --mem=64G
#SBATCH --qos=savio_normal
#SBATCH --output=logs/train_vae_%j.out
#SBATCH --error=logs/train_vae_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu


# Setup:
#   # one-time: featurize PDBs into (C, L, L) .npy stacks consumed by vae.py
#   # bash scripts/featurize_pdb.sh /path/to/ai-cath_subset \
#   #     /global/scratch/users/jqmo/rotation3/datasets/ai-cath_features
#   mkdir -p logs checkpoints
#   sbatch scripts/hpc_scripts/submit_train_savio.sh
#
# Runs train + val + test in one shot:
#   - 70/15/15 split (sklearn.train_test_split, seed=42)
#   - per-epoch validation
#   - best-val checkpoint restored before final test-set evaluation
# Outputs (under $OUTDIR):
#   vae_best.pt        — best-val state_dict + hparams
#   history.json       — per-epoch train/val/recon/kl losses
#   test_metrics.json  — held-out test loss / recon / kl

module load python
module load cuda
# conda activate bioengc242  

FEATURE_DIR="/global/scratch/users/jqmo/rotation3/datasets/ai-cath_features"
OUTDIR="/global/scratch/users/jqmo/rotation3/checkpoints/vae_$(date +%Y%m%d_%H%M%S)"
EPOCHS=100
BATCH_SIZE=64
Z_DIM=64
LR=1e-3
SEED=42

mkdir -p "$OUTDIR"

python /global/scratch/users/jqmo/rotation3/bioengineering242-project/vae/vae.py \
    --feature-dir "$FEATURE_DIR" \
    --outdir "$OUTDIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --z-dim "$Z_DIM" \
    --lr "$LR" \
    --seed "$SEED"
