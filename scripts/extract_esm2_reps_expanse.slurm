#!/bin/bash
#SBATCH -A ucb368
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J esm2_reps
#SBATCH -o logs/esm2_reps_04152026_%j.out
#SBATCH -e logs/esm2_reps_04152026_%j.err
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

set -euo pipefail
export PYTHONUNBUFFERED=1

date
hostname
nvidia-smi

## Load modules
module purge
module load cpu/0.15.4
module load gcc/10.2.0
module load gpu/0.15.4

## paths
REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/bioengc242-project"
PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/cath20-filtered-foldseek"
OUTPUT_BASE="/expanse/lustre/scratch/jmo/temp_project/esm2_embeddings"

mkdir -p "${OUTPUT_BASE}" 

cd "$REPO_ROOT"

## ---------------- conda env ----------------
## On Expanse, `module load anaconda3` does not initialize conda for
## non-interactive shells, so `conda activate` fails. The eval hook
## runs the equivalent of `conda init bash` inline for this session.
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate esm2

python -c "import esm, torch; print('esm OK'); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

## ---------------- run ESM-2 ----------------
echo "==================== ESM-2 reps ===================="
python "${REPO_ROOT}/scripts/extract_esm2_reps.py" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}" \
    --model_name esm2_t36_3B_UR50D \
    --device cuda

## ---------------- end-of-job ----------------
date
echo "Job $SLURM_JOB_ID complete."
if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State || true
fi
