#!/bin/bash
# Run Frame2seq UMAP embedding analysis locally on macOS.
#
# Usage:
#   bash scripts/run_frame2seq_umap_local.sh [INPUT_DIR] [OUTPUT_DIR] [CHAIN_ID]
#
# Defaults assume a local copy of the cath20-filtered-foldseek dataset.

set -euo pipefail

date
echo "Host: $(hostname)"

# paths
INPUT_DIR="/Users/joycemo/Documents/PhD/Rotation3/dataset/switch_proteins"
OUTPUT_DIR="/Users/joycemo/Documents/GitHub/bioengc242-project/output/umap_frame2seek_switch_protein"
CHAIN_ID="${3:-A}"

# env
# with mps
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo ""
echo "=== Environment ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "
import torch
if torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon GPU)')
elif torch.cuda.is_available():
    print('Device: CUDA')
else:
    print('Device: CPU')
"

echo ""
echo "=== Checking dependencies ==="
python -c "import frame2seq; print(f'frame2seq: {frame2seq.__file__}')"
python -c "import umap; print(f'umap-learn: OK')"
python -c "import matplotlib; print(f'matplotlib: OK')"

## mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Running Frame2seq UMAP ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Chain:  $CHAIN_ID"
echo ""

python frame2seq_umap.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --chain-id "$CHAIN_ID" 2>&1

echo ""
echo "Exit code: $?"
echo "=== Done ==="
date
