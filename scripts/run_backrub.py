#!/usr/bin/env python
"""Run Rosetta Backrub ensemble generation on a batch of PDB files.

Interfaces with PyRosetta's BackrubMover protocol.
Designed for use with HPC job arrays — each job processes one PDB.

Usage:
    # Single PDB
    python run_backrub.py --pdb input.pdb --outdir ensembles/backrub --nconfs 5

    # Job array mode
    python run_backrub.py --pdb_list pdb_list.txt --task_id $SLURM_ARRAY_TASK_ID \
                          --outdir ensembles/backrub --nconfs 5
"""

import argparse
import logging
import math
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
        idx = args.task_id - 1  # 1-indexed
        if idx < 0 or idx >= len(lines):
            logger.error(f"task_id {args.task_id} out of range (1-{len(lines)})")
            sys.exit(1)
        return lines[idx]

    logger.error("Provide either --pdb or --pdb_list + --task_id")
    sys.exit(1)


def run_backrub(pdb_path, outdir, n_conformers, n_mc_steps, kT, max_angle, seed):
    """Run Backrub on a single PDB file using PyRosetta."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.backrub import BackrubMover
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.utility import vector1_unsigned_long

    pyrosetta.init(
        "-ignore_unrecognized_res -mute all "
        f"-run:constant_seed -run:jran {seed}",
        set_logging_handler=None,
    )

    stem = Path(pdb_path).stem
    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)

    logger.info(f"Running Backrub on {stem} ({n_conformers} conformers, "
                f"{n_mc_steps} MC steps, kT={kT})")

    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    output_paths = []
    for conf_idx in range(n_conformers):
        work_pose = pose.clone()

        # Configure BackrubMover
        backrub = BackrubMover()
        pivot_residues = vector1_unsigned_long()
        for i in range(1, work_pose.total_residue() + 1):
            if work_pose.residue(i).is_protein():
                pivot_residues.append(i)
        backrub.set_pivot_residues(pivot_residues)
        backrub.set_min_atoms(7)   # ~2 residue segment
        backrub.set_max_atoms(34)  # ~8 residue segment
        backrub.set_max_angle_disp_4(math.radians(max_angle * 0.5))
        backrub.set_max_angle_disp_7(math.radians(max_angle))
        backrub.set_max_angle_disp_slope(0.0)
        backrub.add_mainchain_segments(work_pose)

        # Monte Carlo with Metropolis criterion
        mc = pyrosetta.MonteCarlo(work_pose, scorefxn, kT)
        for _ in range(n_mc_steps):
            backrub.apply(work_pose)
            mc.boltzmann(work_pose)
        mc.recover_low(work_pose)

        out_path = os.path.join(out_sub, f"{stem}_backrub_{conf_idx:03d}.pdb")
        work_pose.dump_pdb(out_path)
        output_paths.append(out_path)
        logger.info(f"  Conformer {conf_idx}: score={scorefxn(work_pose):.1f} -> {out_path}")

    return output_paths


def run_backrub_cli(pdb_path, outdir, n_conformers, n_mc_steps, kT, rosetta_bin,
                    trajectory=False, trajectory_gz=False, trajectory_stride=100):
    """Run Backrub using Rosetta's command-line backrub application."""
    import subprocess

    stem = Path(pdb_path).stem
    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)

    # Protocol: Smith & Kortemme (2008)
    #   10,000 MC trials, 0.6 kT
    #   75% backbone / 25% sidechain moves
    #   Dunbrack backbone-dependent rotamer library
    #   10% of sidechain moves use uniform chi sampling
    #   Retain lowest scoring structure per trajectory
    cmd = [
        rosetta_bin,
        "-s", pdb_path,
        "-backrub:ntrials", str(n_mc_steps),
        "-nstruct", str(n_conformers),
        "-backrub:mc_kt", str(kT),
        "-backrub:sm_prob", "0.25",
        "-backrub:sc_prob_uniform", "0.1",
        "-backrub:sc_prob_withinrot", "0.0",
        "-backrub:initial_pack",
        "-out:path:pdb", out_sub,
        "-out:prefix", f"{stem}_backrub_",
        "-ignore_unrecognized_res",
        "-mute", "all",
    ]

    if trajectory:
        cmd += ["-backrub:trajectory"]
    if trajectory_gz:
        cmd += ["-backrub:trajectory_gz"]
    if trajectory and trajectory_stride:
        cmd += ["-backrub:trajectory_stride", str(trajectory_stride)]

    logger.info(f"Running Rosetta backrub CLI on {stem}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Rosetta backrub failed for {stem}:\n{result.stderr}")
        return []

    outputs = sorted(Path(out_sub).glob(f"{stem}_backrub_*.pdb"))
    logger.info(f"Generated {len(outputs)} conformers -> {out_sub}")
    return [str(p) for p in outputs]


def main():
    parser = argparse.ArgumentParser(description="Rosetta Backrub ensemble generation")
    parser.add_argument("--pdb", type=str, help="Single PDB file path")
    parser.add_argument("--pdb_list", type=str, help="Text file with one PDB path per line")
    parser.add_argument("--task_id", type=int, default=None,
                        help="1-indexed task ID (from $SGE_TASK_ID or $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--nconfs", type=int, default=5, help="Number of conformers (default: 5)")
    parser.add_argument("--nsteps", type=int, default=10000, help="MC steps per conformer (default: 10000)")
    parser.add_argument("--kT", type=float, default=0.6, help="Metropolis kT in kcal/mol (default: 0.6)")
    parser.add_argument("--trajectory", action="store_true",
                        help="Record a trajectory during backrub simulation")
    parser.add_argument("--trajectory_gz", action="store_true",
                        help="Gzip the trajectory output")
    parser.add_argument("--trajectory_stride", type=int, default=100,
                        help="Write a trajectory frame every N steps (default: 100)")
    parser.add_argument("--max_angle", type=float, default=10.0,
                        help="Max backrub rotation angle in degrees (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--mode", choices=["pyrosetta", "cli"], default="cli",
                        help="Use PyRosetta API or Rosetta CLI binary (default: cli)")
    parser.add_argument("--rosetta_bin", type=str, default="backrub",
                        help="Path to Rosetta backrub binary (for --mode cli)")
    args = parser.parse_args()

    pdb_path = get_pdb_path(args)
    if not os.path.isfile(pdb_path):
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "pyrosetta":
        run_backrub(pdb_path, args.outdir, args.nconfs, args.nsteps, args.kT,
                    args.max_angle, args.seed)
    else:
        run_backrub_cli(pdb_path, args.outdir, args.nconfs, args.nsteps, args.kT,
                        args.rosetta_bin, args.trajectory, args.trajectory_gz,
                        args.trajectory_stride)


if __name__ == "__main__":
    main()
