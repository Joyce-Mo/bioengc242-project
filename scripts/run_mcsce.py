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


def preprocess_structure(pdb_path, seed=42):
    """Idealize, repack, and 2-step minimize a structure with HNQ flipping.

    Matches the preprocessing in run_backrub.py:
      1. Idealize bond geometries
      2. Repack all side chains
      3. Enable flip_HNQ
      4. Minimize chi only
      5. Minimize chi + backbone
      6. Disable flip_HNQ

    Returns the path to the preprocessed PDB (written next to the input).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, IncludeCurrent
    from pyrosetta.rosetta.protocols.idealize import IdealizeMover
    from pyrosetta.rosetta.basic.options import set_boolean_option

    pyrosetta.init(
        "-ignore_unrecognized_res -mute all "
        "-ignore_zero_occupancy false "
        "-corrections:beta_nov16 "
        f"-run:constant_seed -run:jran {seed}",
        set_logging_handler=None,
    )

    stem = Path(pdb_path).stem
    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn = ScoreFunctionFactory.create_score_function("beta_nov16")

    logger.info(f"Preprocessing {stem}: initial score {scorefxn(pose):.1f}")

    # 1. Idealize bond geometries
    IdealizeMover().apply(pose)
    logger.info(f"  Post-idealize: {scorefxn(pose):.1f}")

    # 2. Repack side chains
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover()
    packer.score_function(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)
    logger.info(f"  Post-repack: {scorefxn(pose):.1f}")

    # 3. Enable flip_HNQ for minimization
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
    logger.info(f"  Post-minimize (chi only): {scorefxn(pose):.1f}")

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
    logger.info(f"  Post-minimize (chi+bb): {scorefxn(pose):.1f}")

    # 6. Disable flip_HNQ
    set_boolean_option("packing:flip_HNQ", False)

    # Write preprocessed PDB next to original
    out_path = str(Path(pdb_path).parent / f"{stem}_preproc.pdb")
    pose.dump_pdb(out_path)
    logger.info(f"  Preprocessed structure -> {out_path}")
    return out_path


def normalize_pdb_for_amber(pdb_path):
    """Rename atoms and residues from Rosetta conventions to AMBER/MCSCE conventions.

    Operates on fixed-width PDB columns:
      - Atom name:    cols 12-16
      - Residue name: cols 17-20

    Rosetta → AMBER mappings applied:
      Residue names: HIS_D→HID, HSE→HIE, HSP→HIP, HSD→HID, CYX→CYS, MSE→MET
      Atom names:    H1→H at N-terminal (non-PRO); remove H1 for N-terminal PRO;
                     SE→SD for MSE→MET
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    RESNAME_MAP = {
        'HIS_D': 'HID', 'HSD': 'HID', 'HSE': 'HIE', 'HSP': 'HIP',
        'HIS': 'HID',
        'CYX': 'CYS', 'CYD': 'CYS', 'MSE': 'MET',
    }

    out_lines = []
    first_resnum = None
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            out_lines.append(line)
            continue

        atom_name = line[12:16]
        resname = line[17:20].strip()
        resnum = line[22:26].strip()

        # Track first residue number (N-terminal)
        if first_resnum is None:
            first_resnum = resnum

        # Residue name normalization
        if resname in RESNAME_MAP:
            new_resname = RESNAME_MAP[resname]
            # MSE→MET: also rename SE atom to SD
            if resname == 'MSE' and atom_name.strip() == 'SE':
                atom_name = ' SD '
            line = line[:17] + f"{new_resname:>3}" + line[20:]
            line = line[:12] + atom_name + line[16:]

        # N-terminal atom naming
        if resnum == first_resnum and atom_name.strip() == 'H1':
            if resname == 'PRO' or RESNAME_MAP.get(resname) == 'PRO':
                continue  # PRO N-term has no H1 in AMBER
            else:
                atom_name = ' H  '
                line = line[:12] + atom_name + line[16:]

        out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_amber.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    logger.info(f"  Normalized atom/residue names -> {out_path}")
    return out_path


# Standard amino acids recognized by AMBER ff14SB (and their N/C terminal variants)
AMBER_STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLN', 'GLU', 'GLY',
    'HID', 'HIE', 'HIP', 'HYP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
    'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}


def validate_pdb_for_mcsce(pdb_path):
    """Check that all residues in a PDB are standard AMBER residues.

    Returns (is_valid, list_of_bad_residues) where bad_residues contains
    tuples of (resname, resnum, chain_id) for non-standard residues.
    """
    bad_residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            chain = line[21]
            key = (resname, resnum, chain)
            if key in seen:
                continue
            seen.add(key)
            if resname not in AMBER_STANDARD_RESIDUES:
                bad_residues.append(key)
    return len(bad_residues) == 0, bad_residues


def _cleanup_temp_files(pdb_path, preprocess):
    """Remove _amber.pdb and _preproc.pdb temp files."""
    if pdb_path.endswith("_amber.pdb"):
        preproc_path = pdb_path.replace("_amber.pdb", ".pdb")
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
        if preprocess and preproc_path.endswith("_preproc.pdb") and os.path.exists(preproc_path):
            os.remove(preproc_path)


def run_mcsce(pdb_path, outdir, n_conformers, temperature, failed_log=None,
              minimize=True, preprocess=False, seed=42):
    """Run MCSCE on a single PDB file.

    Args:
        preprocess: If True, run idealize + repack + 2-step minimize with
            HNQ flipping on the input structure before MC-SCE.
        minimize: If True, run Rosetta cartesian minimization on each
            generated conformer.
    """
    original_pdb_path = pdb_path
    if preprocess:
        pdb_path = preprocess_structure(pdb_path, seed=seed)

    # Normalize Rosetta atom/residue names to AMBER conventions for MCSCE
    pdb_path = normalize_pdb_for_amber(pdb_path)

    # Pre-filter: skip PDBs with non-standard residues
    is_valid, bad_residues = validate_pdb_for_mcsce(pdb_path)
    if not is_valid:
        bad_str = ", ".join(f"{r[0]}:{r[2]}{r[1]}" for r in bad_residues)
        logger.error(f"SKIPPED {Path(pdb_path).stem}: non-standard residues: {bad_str}")
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{original_pdb_path}\tNON_STANDARD_RESIDUES\t{bad_str}\n")
        _cleanup_temp_files(pdb_path, preprocess)
        return []

    from functools import partial
    from mcsce.libs.libstructure import Structure
    from mcsce.core.side_chain_builder import initialize_func_calc, create_side_chain_ensemble
    from mcsce.core.build_definitions import forcefields
    from mcsce.libs.libenergy import prepare_energy_function

    stem = Path(pdb_path).stem
    # Strip _preproc suffix for output dir naming
    out_stem = stem.replace("_preproc", "")
    out_sub = os.path.join(outdir, out_stem)
    os.makedirs(out_sub, exist_ok=True)

    logger.info(f"Running MC-SCE on {out_stem} ({n_conformers} conformers, T={temperature}K)")

    # Log residue sequence for debugging
    structure = Structure(Path(pdb_path))
    structure.build()
    res_types = list(structure.residue_types)
    logger.info(f"  Residue sequence ({len(res_types)} res): {' '.join(res_types)}")

    try:
        structure = structure.remove_side_chains()

        # Initialize energy calculators (required before ensemble generation)
        ff = forcefields["Amberff14SB"]
        ff_obj = ff(Cterminal='OXT', Nterminal='HN')
        initialize_func_calc(
            partial(prepare_energy_function, batch_size=16,
                    forcefield=ff_obj, terms=["lj", "clash", "coulomb"]),
            structure=structure,
        )

        create_side_chain_ensemble(
            structure=structure,
            n_conformations=n_conformers,
            temperature=temperature,
            save_path=out_sub,
        )
    except (IndexError, ValueError, RuntimeError, KeyError) as e:
        logger.error(f"MC-SCE failed on {out_stem}: {type(e).__name__}: {e}")
        logger.error(f"  Residue sequence was: {' '.join(res_types)}")
        logger.error(f"  N-terminal residue: {res_types[0] if res_types else 'EMPTY'}")
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{original_pdb_path}\t{type(e).__name__}\t{e}\n")
        _cleanup_temp_files(pdb_path, preprocess)
        return []

    _cleanup_temp_files(pdb_path, preprocess)

    outputs = sorted(Path(out_sub).glob("*.pdb"))
    if len(outputs) == 0:
        logger.warning(
            f"Generated 0 conformers for {out_stem} — all {n_conformers} trials "
            f"had unresolvable clashes (all rotamer energies were inf)"
        )
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{original_pdb_path}\tZERO_CONFORMERS\tall trials clashed\n")
    else:
        logger.info(f"Generated {len(outputs)} conformers -> {out_sub}")

    if not minimize or len(outputs) == 0:
        return [str(p) for p in outputs]

    # Rosetta cartesian energy minimization on each conformer
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap

    # Only init if not already initialized by preprocess step
    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init(
            "-ignore_unrecognized_res -mute all "
            "-ignore_zero_occupancy false "
            "-corrections:beta_nov16 "
            f"-run:constant_seed -run:jran {seed}",
            set_logging_handler=None,
        )

    scorefxn_cart = ScoreFunctionFactory.create_score_function("beta_nov16_cart")

    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    mm.set_jump(True)

    min_mover = MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn_cart)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.tolerance(0.01)
    min_mover.cartesian(True)

    minimized_paths = []
    for pdb_out in outputs:
        pose = pyrosetta.pose_from_pdb(str(pdb_out))
        score_before = scorefxn_cart(pose)
        min_mover.apply(pose)
        score_after = scorefxn_cart(pose)
        pose.dump_pdb(str(pdb_out))  # overwrite with minimized structure
        logger.info(f"  Minimized {pdb_out.name}: {score_before:.1f} -> {score_after:.1f}")
        minimized_paths.append(str(pdb_out))

    return minimized_paths


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
    parser.add_argument("--failed_log", type=str, default=None,
                        help="File to append failed PDB paths to")
    parser.add_argument("--no_minimize", action="store_true",
                        help="Skip Rosetta cartesian minimization after MC-SCE")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run idealize + repack + 2-step minimize with HNQ flipping "
                             "on the input structure before MC-SCE (for unminimized inputs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Rosetta (default: 42)")
    args = parser.parse_args()

    pdb_path = get_pdb_path(args)
    if not os.path.isfile(pdb_path):
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_mcsce(pdb_path, args.outdir, args.nconfs, args.temperature,
              failed_log=args.failed_log, minimize=not args.no_minimize,
              preprocess=args.preprocess, seed=args.seed)


if __name__ == "__main__":
    main()
