#!/bin/bash
#$ -S /bin/bash
#$ -N mcsce
#$ -cwd
#$ -t 1-20000  
#$ -tc 200
#$ -l h_rt=00:10:00
#$ -l mem_free=48G
#$ -l scratch=10G
#$ -o logs/mcsce_04242026_$JOB_ID.out
#$ -e logs/mcsce_04242026_$JOB_ID.err
#$ -r y
#$ -m n
#$ -M jqmo@berkeley.edu

# mcsce-precompute backend env: needs openmm, pdbfixer, biopython, tqdm,
# numpy, AND pyrosetta (for the post-MCSCE cartesian min step).
source activate mcsce-precompute

# Tell run_mcsce.py where the mcsce-precompute checkout lives on Wynton.
# run_mcsce.py reads this via os.environ in MCSCE_PRECOMPUTE_DIR (if set);
# otherwise edit MCSCE_PRECOMPUTE_DIR at the top of run_mcsce.py.
export MCSCE_PRECOMPUTE_DIR="/wynton/home/rotation/jqmo/rotation3/mcsce-precompute"

# /wynton/scratch/jqmo/rotation_datasets/augmented_ingraham_cath_bugfree/mpnn_esmfold 

PDB_LIST="/wynton/scratch/jqmo/rotation_datasets/ai_cath_training_filtered/kept_pdb_list.txt"
OUTDIR="/wynton/scratch/jqmo/rotation_datasets/final_ai-cath_mcsce_ensembles"
NCONFS=100
FAILED_LOG="logs/mcsce_failed_pdbs.txt"

python scripts/run_mcsce.py \
    --pdb_list "$PDB_LIST" \
    --task_id "$SGE_TASK_ID" \
    --outdir "$OUTDIR" \
    --nconfs "$NCONFS" \
    --failed_log "$FAILED_LOG" \
    --device CPU \
    --num_workers 1

