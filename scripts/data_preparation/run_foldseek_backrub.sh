#!/bin/bash
# Run foldseek structural analysis on the ai-cath_subset dataset.
#
# Compares:
#   1. Originals only (all-vs-all)
#   2. Originals + Backrub conformers (all-vs-all, to see how ensembles
#      cluster relative to their parent structures)
#
# Uses foldseek_analysis.py from the protein_augmentation repo.
#
# Usage:
#   bash scripts/run_foldseek_backrub.sh

set -euo pipefail

DATA_DIR="/Users/joycemo/Documents/PhD/Rotation3/dataset/ai-cath_subset"
FOLDSEEK_SCRIPT="/Users/joycemo/Documents/GitHub/protein_augmentation/scripts/foldseek_analysis.py"
OUTPUT_BASE="/Users/joycemo/Documents/PhD/Rotation3/dataset/ai-cath_subset/analysis/"

# --------------------------------------------------------------------------
# 1. Foldseek on originals only
# --------------------------------------------------------------------------
echo "=== Step 1: Foldseek analysis on original structures ==="
python "$FOLDSEEK_SCRIPT" \
    --pdb-dir "${DATA_DIR}/originals" \
    --output-dir "${OUTPUT_BASE}/originals" \
    --run-foldseek \
    --threads 4

# --------------------------------------------------------------------------
# 2. Stage a flat directory with originals + all backrub conformers
#    (foldseek needs all PDBs in one directory)
# --------------------------------------------------------------------------
echo "=== Step 2: Staging combined directory ==="
COMBINED_DIR="${OUTPUT_BASE}/combined_pdbs"
mkdir -p "$COMBINED_DIR"

# Copy originals (prefix with "orig_" to distinguish in results)
for pdb in "${DATA_DIR}"/originals/*.pdb; do
    stem=$(basename "$pdb" .pdb)
    cp "$pdb" "${COMBINED_DIR}/${stem}.pdb"
done

# Copy backrub conformers
for conf_dir in "${DATA_DIR}"/backrub_ensembles/*/; do
    for pdb in "${conf_dir}"*.pdb; do
        [ -f "$pdb" ] && cp "$pdb" "${COMBINED_DIR}/"
    done
done

echo "  Combined directory: $(ls "${COMBINED_DIR}"/*.pdb | wc -l) PDB files"

# --------------------------------------------------------------------------
# 3. Foldseek on combined originals + backrub conformers
# --------------------------------------------------------------------------
echo "=== Step 3: Foldseek analysis on originals + backrub conformers ==="
python "$FOLDSEEK_SCRIPT" \
    --pdb-dir "$COMBINED_DIR" \
    --output-dir "${OUTPUT_BASE}/combined" \
    --run-foldseek \
    --threads 4

# --------------------------------------------------------------------------
# 4. RMSD analysis (backrub vs originals)
# --------------------------------------------------------------------------
echo "=== Step 4: RMSD analysis (backrub conformers vs originals) ==="
python scripts/rmsd_backrub_vs_original.py \
    --data-dir "$DATA_DIR" \
    --output-dir "${OUTPUT_BASE}/rmsd"

echo ""
echo "=== All done! ==="
echo "  Originals foldseek:  ${OUTPUT_BASE}/originals/"
echo "  Combined foldseek:   ${OUTPUT_BASE}/combined/"
echo "  RMSD analysis:       ${OUTPUT_BASE}/rmsd/"
