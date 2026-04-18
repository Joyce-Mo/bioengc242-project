#!/bin/bash
#SBATCH -A ucb368
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J esmfold_reps
#SBATCH -o logs/esmfold_reps_04152026_%j.out
#SBATCH -e logs/esmfold_reps_04152026_%j.err
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# ESMFold is substantially larger than ESM-2 alone (ESM-2 3B + folding trunk),
# and the pairwise z representation is quadratic in sequence length.
# Memory request bumped to 128G from 64G, wall time to 24h from 12h.

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
OUTPUT_BASE="/expanse/lustre/scratch/jmo/temp_project/esmfold_embeddings"

mkdir -p "${OUTPUT_BASE}"

cd "$REPO_ROOT"

## conda env
## On Expanse, `module load anaconda3` does not initialize conda for
## non-interactive shells, so `conda activate` fails. The eval hook
## runs the equivalent of `conda init bash` inline for this session.
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate esm2

python -c "import esm, torch; print('esm OK'); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

## run ESMFold extraction
echo "==================== ESMFold reps ===================="
python "${REPO_ROOT}/scripts/extract_esmfold_reps.py" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}" \
    --device cuda \
    --max_len 1024

## end of job
date
echo "Job $SLURM_JOB_ID complete."
if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State || true
fi
