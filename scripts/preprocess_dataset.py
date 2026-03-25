#!/usr/bin/env python
"""Batch preprocess PDB files into torch_geometric graph .pt files.

Usage:
    # Process all PDBs
    python preprocess_dataset.py --pdb_dir /path/to/pdbs --out_dir /path/to/processed

    # Job array mode (one PDB per task)
    python preprocess_dataset.py --pdb_list pdb_list.txt --task_id $SLURM_ARRAY_TASK_ID \
                                  --out_dir /path/to/processed
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.graph_builder import pdb_to_graph


def preprocess_single(pdb_path: str, out_dir: str, knn_k: int = 30):
    stem = Path(pdb_path).stem
    out_path = os.path.join(out_dir, f"{stem}.pt")

    if os.path.exists(out_path):
        logger.info(f"Skipping {stem} (already processed)")
        return

    try:
        data = pdb_to_graph(pdb_path, k=knn_k)
        data.name = stem
        torch.save(data, out_path)
        logger.info(f"Processed {stem} -> {out_path}")
    except Exception as e:
        logger.warning(f"Failed {stem}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, help="Directory of PDB files (sequential mode)")
    parser.add_argument("--pdb_list", type=str, help="File listing PDB paths (array mode)")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--knn_k", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pdb_list and args.task_id is not None:
        # Array mode: process single PDB
        with open(args.pdb_list) as f:
            lines = [l.strip() for l in f if l.strip()]
        idx = args.task_id - 1
        if idx < 0 or idx >= len(lines):
            logger.error(f"task_id {args.task_id} out of range")
            sys.exit(1)
        preprocess_single(lines[idx], args.out_dir, args.knn_k)

    elif args.pdb_dir:
        # Sequential mode: process all PDBs
        from tqdm import tqdm
        pdb_files = sorted(Path(args.pdb_dir).glob("*.pdb"))
        for pdb_path in tqdm(pdb_files, desc="Preprocessing"):
            preprocess_single(str(pdb_path), args.out_dir, args.knn_k)
    else:
        logger.error("Provide --pdb_dir or --pdb_list + --task_id")
        sys.exit(1)


if __name__ == "__main__":
    main()
