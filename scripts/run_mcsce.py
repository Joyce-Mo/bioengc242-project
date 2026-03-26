#!/usr/bin/env python
"""Run MC-SCE ensemble generation on a batch of PDB files.

Interfaces with the MCSCE package (github.com/THGLab/MCSCE).
Designed for use with HPC job arrays — each job processes one PDB.

Usage:
    # Single PDB
    python run_mcsce.py --pdb input.pdb --outdir ensembles/mcsce --nconfs 5

    # Job array mode: process the PDB at line $TASK_ID in the file list
    python run_mcsce.py --pdb_list pdb_list.txt --task_id $SGE_TASK_ID \
                        --outdir ensembles/mcsce --nconfs 5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_pdb_path(args):
    """Resolve the PDB path from either --pdb or --pdb_list + --task_id."""
    if args.pdb:
        return args.pdb

    if args.pdb_list and args.task_id is not None:
        with open(args.pdb_list) as f:
            lines = [l.strip() for l in f if l.strip()]
        idx = args.task_id - 1  # job arrays are 1-indexed
        if idx < 0 or idx >= len(lines):
            logger.error(f"task_id {args.task_id} out of range (1-{len(lines)})")
            sys.exit(1)
        return lines[idx]

    logger.error("Provide either --pdb or --pdb_list + --task_id")
    sys.exit(1)


def run_mcsce(pdb_path, outdir, n_conformers, temperature, n_trials, failed_log=None):
    """Run MCSCE on a single PDB file."""
    from mcsce.core.side_chain_builder import create_side_chain_ensemble

    stem = Path(pdb_path).stem
    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)

    logger.info(f"Running MC-SCE on {stem} ({n_conformers} conformers, T={temperature}K)")

    try:
        create_side_chain_ensemble(
            input_pdb=pdb_path,
            n_conf=n_conformers,
            output_folder=out_sub,
            temperature=temperature,
            n_trials=n_trials,
        )
    except (IndexError, ValueError) as e:
        logger.error(f"MC-SCE failed on {stem}: {e}")
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{pdb_path}\n")
        return []

    outputs = sorted(Path(out_sub).glob("*.pdb"))
    logger.info(f"Generated {len(outputs)} conformers -> {out_sub}")
    return [str(p) for p in outputs]


def main():
    parser = argparse.ArgumentParser(description="MC-SCE ensemble generation")
    parser.add_argument("--pdb", type=str, help="Single PDB file path")
    parser.add_argument("--pdb_list", type=str, help="Text file with one PDB path per line")
    parser.add_argument("--task_id", type=int, default=None,
                        help="1-indexed task ID (from $SGE_TASK_ID or $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--nconfs", type=int, default=5, help="Number of conformers (default: 5)")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Sampling temperature in Kelvin (default: 300)")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Rosenbluth trial moves per residue (default: 10)")
    parser.add_argument("--failed_log", type=str, default=None,
                        help="File to append failed PDB paths to")
    args = parser.parse_args()

    pdb_path = get_pdb_path(args)
    if not os.path.isfile(pdb_path):
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_mcsce(pdb_path, args.outdir, args.nconfs, args.temperature,
              args.n_trials, args.failed_log)


if __name__ == "__main__":
    main()
