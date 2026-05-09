#!/bin/bash
#SBATCH -A ucb368
#SBATCH --job-name=tar_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH -o logs/tar_embeddings_%j.out
#SBATCH -e logs/tar_embeddings_%j.err
#SBATCH -p shared
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

# Create tar.gz archives of embedding directories.
# Runs as a batch job so it won't die when SSH disconnects.

set -euo pipefail

cd /expanse/lustre/scratch/jmo/temp_project

echo "Starting tar jobs at $(date)"

# Boltz2 embeddings
echo "Archiving boltz2_embeddings..."
tar -czf boltz2_embeddings.tar.gz boltz2_embeddings/
echo "Done boltz2_embeddings at $(date)"

# ESM2 embeddings
echo "Archiving esm2_embeddings..."
tar -czf esm2_embeddings.tar.gz esm2_embeddings/
echo "Done esm2_embeddings at $(date)"

# ESMFold embeddings
echo "Archiving esmfold_embeddings..."
tar -czf esmfold_embeddings.tar.gz esmfold_embeddings/
echo "Done esmfold_embeddings at $(date)"

echo "All archives complete at $(date)"
ls -lh *.tar.gz
