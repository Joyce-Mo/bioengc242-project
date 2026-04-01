#!/bin/bash
#$ -S /bin/bash
#$ -N mcsce
#$ -cwd
#$ -t 1-30
#$ -tc 200
#$ -l h_rt=15:00:00
#$ -l mem_free=48G
#$ -l scratch=10G
#$ -o logs/mcsce_04012026_$JOB_ID.out
#$ -e logs/mcsce_04012026_$JOB_ID.err
#$ -r y
#$ -m n
#$ -M jqmo@berkeley.edu

source activate mcsce

PDB_LIST="pdb_list_ai-cath_subset.txt"
OUTDIR="/wynton/scratch/jqmo/rotation_datasets/ai-cath_mcsce_ensembles"
NCONFS=10
FAILED_LOG="logs/mcsce_failed_pdbs.txt"

python scripts/run_mcsce.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SGE_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --failed_log "$FAILED_LOG" \
    --preprocess

