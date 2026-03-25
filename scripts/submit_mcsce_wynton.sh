#!/bin/bash
#$ -S /bin/bash
#$ -N mcsce
#$ -cwd
#$ -t 1-13129
#$ -tc 200
#$ -l h_rt=15:00:00
#$ -l mem_free=48G
#$ -l scratch=10G
#$ -o logs/mcsce_$JOB_ID.out
#$ -e logs/mcsce_$JOB_ID.err
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu
#
# MC-SCE ensemble generation on Wynton (SGE)
#
# Setup:
#   bash make_pdb_list.sh /path/to/cath20-filtered-foldseek > pdb_list.txt
#   mkdir -p logs
#   qsub submit_mcsce_wynton.sh

# conda activate mcsce

PDB_LIST="pdb_list.txt"
OUTDIR="ensembles/mcsce"
NCONFS=5
TEMPERATURE=300.0

python run_mcsce.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SGE_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --temperature "$TEMPERATURE"
