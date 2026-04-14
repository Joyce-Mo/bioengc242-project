#!/bin/bash
#$ -S /bin/bash
#$ -N filter_ai-cath
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l mem_free=16G
#$ -o logs/filter_ai-cath_$JOB_ID_04142026.out
#$ -e logs/filter_ai-cath_$JOB_ID_04142026.err
#$ -m b,e
#$ -M jqmo@berkeley.edu

source activate mcsce

# ── Mode selection 
# Set MODE=pdb_list to validate the flat training dataset (pre-backrub).
# Set MODE=ensemble to filter backrub conformers vs originals (post-backrub).
MODE="${MODE:-pdb_list}"

if [ "$MODE" = "pdb_list" ]; then
    # Validate the ai-cath training PDB dataset
    PDB_LIST="ai-cath_training_pdb.txt"
    ORIGINALS_DIR="/wynton/scratch/jqmo/rotation_datasets/OG_ingraham_cath"
    OUTPUT_DIR="/wynton/scratch/jqmo/rotation_datasets/ai_cath_training_filtered"

    python scripts/data_preparation/filter_backrub_quality.py \
        --pdb-list "$PDB_LIST" \
        --originals-dir "$ORIGINALS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --score-min -1200 \
        --score-max -0 \
        --rmsd-max 2.0

elif [ "$MODE" = "ensemble" ]; then
    # Filter backrub ensembles vs originals
    DATA_DIR="/wynton/scratch/jqmo/rotation_datasets"
    OUTPUT_DIR="/wynton/scratch/jqmo/rotation_datasets/ai_cath_backrub_filtered"
    ENSEMBLE_DIR="/wynton/scratch/jqmo/rotation_datasets/ai_cath_backrub_redo_1"

    python scripts/data_preparation/filter_backrub_quality.py \
        --data-dir "$DATA_DIR" \
        --ensemble-dir "$ENSEMBLE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --score-min -1200 \
        --score-max -0 \
        --drmsd-max 5.0
fi
