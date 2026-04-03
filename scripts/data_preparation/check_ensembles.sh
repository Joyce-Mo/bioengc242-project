#!/bin/bash
# Check ensemble output directories for PDB files
# Usage: bash scripts/check_ensembles.sh /path/to/ensembles_dir

ENSDIR="${1:?Usage: bash scripts/check_ensembles.sh /path/to/ensembles_dir}"

total=0
empty=0
nonempty=0
empty_list=""

for subdir in "$ENSDIR"/*/; do
    [ -d "$subdir" ] || continue
    total=$((total + 1))
    name=$(basename "$subdir")
    count=$(find "$subdir" -maxdepth 1 -name "*.pdb" -type f | wc -l)
    if [ "$count" -eq 0 ]; then
        empty=$((empty + 1))
        empty_list="${empty_list}${name}\n"
    else
        nonempty=$((nonempty + 1))
        echo "[OK]    $name: $count PDBs"
    fi
done

echo ""
echo "=== Summary ==="
echo "Total subdirectories: $total"
echo "With PDBs:           $nonempty"
echo "Empty:               $empty"

if [ "$empty" -gt 0 ]; then
    echo ""
    echo "=== Empty directories ==="
    echo -e "$empty_list"
fi

# Also check if any PDBs landed directly in the top-level dir
top_pdbs=$(find "$ENSDIR" -maxdepth 1 -name "*.pdb" -type f | wc -l)
if [ "$top_pdbs" -gt 0 ]; then
    echo ""
    echo "WARNING: $top_pdbs PDB files found in top-level directory (not in subdirs)"
fi
