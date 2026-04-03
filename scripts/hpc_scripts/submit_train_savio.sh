#!/bin/bash
#SBATCH --job-name=train_vae_diff
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Training on Savio GPU node
#
# Phase 1 (VAE pre-training):
#   sbatch submit_train_savio.sh --phase 1
#
# Phase 2 (joint VAE + diffusion):
#   sbatch submit_train_savio.sh --phase 2 --vae_ckpt checkpoints/vae_best.pt

module load python
# conda activate your_env

PHASE="${1:-1}"
VAE_CKPT="${2:-}"
CONFIG="../configs/default.yaml"

cd "$(dirname "$0")/.."

if [ "$PHASE" = "1" ]; then
    python -m src.training.train_vae --config "$CONFIG"
elif [ "$PHASE" = "2" ]; then
    EXTRA=""
    if [ -n "$VAE_CKPT" ]; then
        EXTRA="--vae_checkpoint $VAE_CKPT"
    fi
    python -m src.training.train_joint --config "$CONFIG" $EXTRA
fi
