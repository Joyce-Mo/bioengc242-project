#!/bin/bash
#$ -S /bin/bash
#$ -N mcsce
#$ -cwd
#$ -t 1-30
#$ -tc 200
#$ -l h_rt=15:00:00
#$ -l mem_free=48G
#$ -l scratch=10G
#$ -o logs/mcsce_03252026$JOB_ID.out
#$ -e logs/mcsce_03252026$JOB_ID.err
#$ -r y
#$ -m n
#$ -M joyce.mo@ucsf.edu
#

source activate mcsce

PDB_LIST="pdb_list_ai-cath_subset.txt"
OUTDIR="/wynton/scratch/jqmo/rotation_datasets/ai-cath_mcsce_ensembles"
NCONFS=100

# Get the PDB path for this array task
PDB_PATH=$(sed -n "${SGE_TASK_ID}p" "$PDB_LIST")
STEM=$(basename "$PDB_PATH" .pdb)

FAILED_LOG="logs/mcsce_failed_pdbs.txt"

mcsce "$PDB_PATH" "$NCONFS" \
    -o "${OUTDIR}/${STEM}" \
    -m ensemble

if [ $? -ne 0 ]; then
    echo "$PDB_PATH" >> "$FAILED_LOG"
fi
