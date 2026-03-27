#!/bin/bash
#$ -S /bin/bash
#$ -N backrub
#$ -cwd
#$ -t 1-30
#$ -tc 200
#$ -l h_rt=04:00:00
#$ -l mem_free=8G
#$ -o logs/backrub_03272026_$JOB_ID.out
#$ -e logs/backrub_03272026_$JOB_ID.err

source activate mcsce

PDB_LIST="pdb_list_ai-cath_subset.txt"
OUTDIR="/wynton/scratch/jqmo/rotation_datasets/ai-cath_backrub_subset_ensembles"
NCONFS=5
NSTEPS=10000
KT=0.6

python scripts/run_backrub.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SGE_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --nsteps "$NSTEPS" \
    --kT "$KT" \
    --mode pyrosetta
