#!/usr/bin/env python
"""Run Rosetta Backrub ensemble generation on a batch of PDB files.

Interfaces with PyRosetta's BackrubMover protocol.
Designed for use with HPC job arrays — each job processes one PDB.

Usage:
    # Single PDB
    python run_backrub.py --pdb input.pdb --outdir ensembles/backrub --nconfs 5

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


def preprocess_pose(pose, scorefxn):
    """Preprocess a pose: idealize, repack, and two-stage minimize.

    Shared preprocessing protocol used before backrub sampling and for
    energy-based quality filtering. Modifies the pose in place.

    Protocol (adapted from Smith & Kortemme 2008/2010):
      1. Idealize bond geometries (IdealizeMover, Engh & Huber values)
      2. Repack side chains (PackRotamersMover with RestrictToRepacking + IncludeCurrent)
      3. Enable flip_HNQ for His/Asn/Gln sidechain flips during minimization
      4. L-BFGS minimize chi angles only
      5. L-BFGS minimize chi + backbone
      6. Disable flip_HNQ

    Args:
        pose: PyRosetta Pose object to preprocess.
        scorefxn: Rosetta ScoreFunction (beta_nov16).

    Returns:
        The minimized score (REU).
    """
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover, PackRotamersMover,
    )
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking, IncludeCurrent,
    )
    from pyrosetta.rosetta.protocols.idealize import IdealizeMover
    from pyrosetta.rosetta.basic.options import set_boolean_option

    # 1. Idealize bond geometries before repacking
    idealize = IdealizeMover()
    idealize.apply(pose)
    logger.info(f"Post-idealize score: {scorefxn(pose):.1f}")

    # 2. Repack side chains before minimization
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover()
    packer.score_function(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)
    logger.info(f"Post-repack score: {scorefxn(pose):.1f}")

    # 3. Enable flip_HNQ for minimization (Dru suggestion)
    set_boolean_option("packing:flip_HNQ", True)

    # 4. Minimize side chains only
    mm_chi = MoveMap()
    mm_chi.set_bb(False)
    mm_chi.set_chi(True)
    min_chi = MinMover()
    min_chi.movemap(mm_chi)
    min_chi.score_function(scorefxn)
    min_chi.min_type("lbfgs_armijo_nonmonotone")
    min_chi.tolerance(0.01)
    min_chi.apply(pose)
    logger.info(f"Post-minimize (chi only): {scorefxn(pose):.1f}")

    # 5. Minimize side chains + backbone
    mm_all = MoveMap()
    mm_all.set_bb(True)
    mm_all.set_chi(True)
    min_all = MinMover()
    min_all.movemap(mm_all)
    min_all.score_function(scorefxn)
    min_all.min_type("lbfgs_armijo_nonmonotone")
    min_all.tolerance(0.01)
    min_all.apply(pose)
    logger.info(f"Post-minimize (chi+bb): {scorefxn(pose):.1f}")

    # 6. Disable flip_HNQ for subsequent steps
    set_boolean_option("packing:flip_HNQ", False)

    return scorefxn(pose)


def run_backrub(pdb_path, outdir, n_conformers, n_mc_steps, kT, max_angle, seed,
                repack=False, trajectory_stride=100):
    """Run Backrub on a single PDB following Smith & Kortemme (2010) paper's methods


    Overview of Protocol from Smith & Kortemme (2010) paper:
      1. Repack all side chains via MC simulated annealing
      2. Two-stage minimization: (a) side chains only, (b) side chains + backbone
      3. Backrub MC with 75% backbone / 25% sidechain moves, Dunbrack rotamers,
         10% uniform chi sampling. Retains lowest-scoring structure per trajectory.
      4. Post-backrub two-stage minimization on each conformer

       Although note that I did update protocol for...
    - newer score function
    - minimization of side chain, separately, followed by side chain + bacbone (instead of
    some random backbone residues near mutations of the PDZ that Kortemme paper was looking at)
    - idealization of bond geometries during repacking step
    - explicitly flip_HNQ during minimization but not during backrub sampling


    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap

    pyrosetta.init(
        "-ignore_unrecognized_res -mute all "
        "-ignore_zero_occupancy false "
        "-corrections:beta_nov16 "
        f"-run:constant_seed -run:jran {seed}",
        set_logging_handler=None,
    )

    stem = Path(pdb_path).stem
    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)

    logger.info(f"Running PyRosetta on {stem} ({n_conformers} conformers, "
                f"{n_mc_steps} MC steps, kT={kT})")

    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn = ScoreFunctionFactory.create_score_function("beta_nov16")

    initial_score = scorefxn(pose)
    logger.info(f"  Initial score: {initial_score:.1f}")

    # Shared preprocessing: idealize, repack, two-stage minimize
    preprocess_pose(pose, scorefxn)

    # Set up minimization movers for post-backrub minimization of each conformer
    mm_chi = MoveMap()
    mm_chi.set_bb(False)
    mm_chi.set_chi(True)
    min_chi = MinMover()
    min_chi.movemap(mm_chi)
    min_chi.score_function(scorefxn)
    min_chi.min_type("lbfgs_armijo_nonmonotone")
    min_chi.tolerance(0.01)

    mm_all = MoveMap()
    mm_all.set_bb(True)
    mm_all.set_chi(True)
    min_all = MinMover()
    min_all.movemap(mm_all)
    min_all.score_function(scorefxn)
    min_all.min_type("lbfgs_armijo_nonmonotone")
    min_all.tolerance(0.01)

    # Backrub MC sampling
    # Smith & Kortemme (2010): backbone backrub moves, retain lowest-scoring.
    from pyrosetta.rosetta.protocols.backrub import BackrubMover
    from pyrosetta.rosetta.protocols.moves import MonteCarlo

    backrub_mover = BackrubMover()
    backrub_mover.set_max_angle_disp_4(max_angle)
    backrub_mover.set_max_angle_disp_7(max_angle)

    output_paths = []
    for conf_idx in range(n_conformers):
        work_pose = pose.clone()

        # Set up backrub segments on the work pose
        backrub_mover.set_input_pose(work_pose)
        backrub_mover.clear_segments()
        backrub_mover.add_mainchain_segments_from_options(work_pose)
        logger.info(f"  Conformer {conf_idx}: {backrub_mover.num_segments()} backrub segments")

        mc = MonteCarlo(work_pose, scorefxn, kT)

        # Track energy trajectory during MC sampling
        trajectory = []
        for step in range(n_mc_steps):
            backrub_mover.apply(work_pose)
            accepted = mc.boltzmann(work_pose)
            if step % trajectory_stride == 0:
                trajectory.append({
                    "step": step,
                    "score_current": scorefxn(work_pose),
                    "score_lowest": mc.lowest_score(),
                    "accepted": accepted,
                })
        mc.recover_low(work_pose)
        score_after_mc = scorefxn(work_pose)

        # Post-backrub two-stage minimization on each conformer
        min_chi.apply(work_pose)
        score_after_min_chi = scorefxn(work_pose)
        min_all.apply(work_pose)
        score_after_min_all = scorefxn(work_pose)

        # Append minimization stages to trajectory
        trajectory.append({"step": "post_mc_lowest", "score_current": score_after_mc,
                           "score_lowest": score_after_mc, "accepted": True})
        trajectory.append({"step": "post_min_chi", "score_current": score_after_min_chi,
                           "score_lowest": score_after_min_chi, "accepted": True})
        trajectory.append({"step": "post_min_all", "score_current": score_after_min_all,
                           "score_lowest": score_after_min_all, "accepted": True})

        # Save trajectory CSV
        import csv
        traj_path = os.path.join(out_sub, f"trajectory_{conf_idx:03d}.csv")
        with open(traj_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", "score_current", "score_lowest", "accepted"])
            writer.writeheader()
            writer.writerows(trajectory)

        out_path = os.path.join(out_sub, f"{stem}_backrub_{conf_idx:03d}.pdb")
        work_pose.dump_pdb(out_path)
        output_paths.append(out_path)
        logger.info(f"  Conformer {conf_idx}: MC={score_after_mc:.1f} -> "
                     f"min_chi={score_after_min_chi:.1f} -> "
                     f"min_all={score_after_min_all:.1f} -> {out_path}")

    return output_paths


def run_backrub_cli(pdb_path, outdir, n_conformers, n_mc_steps, kT, rosetta_bin,
                    trajectory=False, trajectory_gz=False, trajectory_stride=100):
    """Run Backrub using Rosetta's command-line backrub application."""
    import subprocess

    stem = Path(pdb_path).stem
    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)

    # Protocol: Smith & Kortemme (2010)
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
    parser.add_argument("--repack", action="store_true",
                        help="Repack all side chains (MC simulated annealing) before minimization")
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
                    args.max_angle, args.seed, repack=args.repack,
                    trajectory_stride=args.trajectory_stride)
    else:
        run_backrub_cli(pdb_path, args.outdir, args.nconfs, args.nsteps, args.kT,
                        args.rosetta_bin, args.trajectory, args.trajectory_gz,
                        args.trajectory_stride)


if __name__ == "__main__":
    main()
