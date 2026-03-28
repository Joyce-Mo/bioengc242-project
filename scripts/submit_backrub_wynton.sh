#!/bin/bash
#$ -S /bin/bash
#$ -N backrub
#$ -cwd
#$ -t 1-100000
#$ -tc 200
#$ -l h_rt=10:00:00
#$ -l mem_free=8G
#$ -o logs/backrub_03272026_$JOB_ID.out
#$ -e logs/backrub_03272026_$JOB_ID.err

source activate mcsce

PDB_LIST="pdb_list.txt"
OUTDIR="/wynton/scratch/jqmo/rotation_datasets/ai_cath_backrub_all"
SUBSET_DIR="/wynton/scratch/jqmo/rotation_datasets/ai-cath_backrub_subset_ensembles"
NCONFS=3
NSTEPS=10000
KT=0.6

# Get the PDB path for this task and extract the stem (filename without extension)
PDB_PATH=$(sed -n "${SGE_TASK_ID}p" "$PDB_LIST")
PDB_STEM=$(basename "$PDB_PATH" .pdb)

# Skip if backrub conformations already exist in subset folder
if [ -d "${SUBSET_DIR}/${PDB_STEM}" ] && [ "$(ls -A "${SUBSET_DIR}/${PDB_STEM}"/*.pdb 2>/dev/null)" ]; then
    echo "Skipping ${PDB_STEM}: conformations already exist in ${SUBSET_DIR}/${PDB_STEM}"
    exit 0
fi

python scripts/run_backrub.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SGE_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --nsteps "$NSTEPS" \
    --kT "$KT" \
    --mode pyrosetta
