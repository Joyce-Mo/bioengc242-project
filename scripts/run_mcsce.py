#!/usr/bin/env python
"""Run MC-SCE ensemble generation on a batch of PDB files.

Interfaces with MCSCE cloned form teresa's lab (github.com/THGLab/MCSCE).
Designed for use with HPC job arrays. Each job processes one PDB. 

Usage:
python run_mcsce.py --pdb input.pdb --outdir ensembles/mcsce --nconfs 5

Make sure to run from mcsce env! 
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

# Force unbuffered output — ensures HPC logs are written before crashes
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


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


def minimize_structure(pdb_path, seed=42):
    """Idealize, repack, and 2-step minimize a structure with HNQ flipping.

    Matches the preprocessing protocol from run_backrub.py (Smith & Kortemme, 2010):
      1. Idealize bond geometries
      2. Repack side chains (MC simulated annealing)
      3. Minimize chi only (lbfgs, tol=0.01)
      4. Minimize chi + backbone (lbfgs, tol=0.01)

    Returns the path to the preprocessed PDB (written next to the input).
    """

    # note the following 50 or so lines with rosetta are just copied and pasted from run_backrub
    # TODO: for better code reuse, could turns the rosetta minimization in a separate function and call
    # but i am feelign lazy rn and will just copy and paste.
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover, PackRotamersMover,
    )
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, IncludeCurrent
    from pyrosetta.rosetta.protocols.idealize import IdealizeMover
    from pyrosetta.rosetta.basic.options import set_boolean_option

    # Only init PyRosetta if not already initialized (e.g. by a prior step)
    if not pyrosetta.rosetta.basic.was_init_called():
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

    initial_score = scorefxn(pose)
    logger.info(f"  Initial score: {initial_score:.1f}")

    # Idealize bond geometries before repacking
    idealize = IdealizeMover()
    idealize.apply(pose)
    logger.info(f"  Post-idealize score: {scorefxn(pose):.1f}")

    # Repack side chains before minimization and backrub sampling
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover()
    packer.score_function(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)
    logger.info(f"  Post-repack score: {scorefxn(pose):.1f}")

    # Enable flip_HNQ for minimization (Dru suggestion)
    set_boolean_option("packing:flip_HNQ", True)

    # Step 1: Minimize side chains only
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

    # Stage 2: Minimize side chains + backbone
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

    # Disable flip_HNQ for backrub sampling and remainder
    set_boolean_option("packing:flip_HNQ", False)

    # Write preprocessed PDB next to original
    out_path = str(Path(pdb_path).parent / f"{stem}_preproc.pdb")
    pose.dump_pdb(out_path)
    logger.info(f"  Preprocessed structure -> {out_path}")
    return out_path

def remove_sidechain(pdb_path):
    """Remove side chain atoms from a PDB, retaining backbone + CB per MC-SCE paper.

    Per the MC-SCE method (Bhatt & Bhatt):
      "all the side chain atoms, except the Cβ atom, and any existing water
       molecules are eliminated."

    Retained atoms: N, CA, C, O, CB, OXT, and backbone hydrogens (H, H1, H2, H3).
    GLY has no CB — only backbone atoms are kept.
    Water molecules (HOH, WAT, TIP3, SOL) are removed.

    Note: backbone H atoms must be kept for MCSCE's energy calculations
    (they are part of MCSCE's internal backbone_atoms definition).

    Operates on fixed-width PDB columns:
      - Atom name:    cols 12-16
      - Residue name: cols 17-20
    """
    # Backbone atoms to keep — must match MCSCE's backbone_atoms definition
    # which is ('N', 'C', 'CA', 'O', 'OXT', 'H', 'H1', 'H2', 'H3')
    # plus CB per MC-SCE paper
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'CB', 'OXT', 'H', 'H1', 'H2', 'H3'}
    # Water residue names across common conventions
    WATER_RESNAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'TIP'}

    with open(pdb_path) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            resname = line[17:20].strip()
            # Remove water molecules
            if resname in WATER_RESNAMES:
                continue
            atom_name = line[12:16].strip()
            # Keep backbone atoms + CB only
            if atom_name in BACKBONE_ATOMS:
                out_lines.append(line)
        else:
            out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_bb.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    logger.info(f"  Removed side chains (kept CB) -> {out_path}")
    return out_path


def normalize_pdb_for_amber(pdb_path):
    """Rename atoms and residues from Rosetta conventions to AMBER/MCSCE conventions.

    Operates on fixed-width PDB columns:
      - Atom name:    cols 12-16
      - Residue name: cols 17-20

    Rosetta → AMBER mappings applied:
      Residue names: HIS_D→HID, HSE→HIE, HSP→HIP, HSD→HID, CYX→CYS, MSE→MET
      Atom names:    SE→SD for MSE→MET

    Note: N-terminal H1/H2/H3 atoms are NOT renamed — MCSCE's internal code
    (get_all_backbone_atom_coords) expects exactly H1, H2, H3 at the N-terminal.
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    RESNAME_MAP = {
        'HIS_D': 'HID', 'HSD': 'HID', 'HSE': 'HIE', 'HSP': 'HIP',
        'HIS': 'HID',
        'CYX': 'CYS', 'CYD': 'CYS', 'MSE': 'MET',
    }

    out_lines = []
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            out_lines.append(line)
            continue

        atom_name = line[12:16]
        resname = line[17:20].strip()

        # Residue name normalization
        if resname in RESNAME_MAP:
            new_resname = RESNAME_MAP[resname]
            # MSE→MET: also rename SE atom to SD
            if resname == 'MSE' and atom_name.strip() == 'SE':
                atom_name = ' SD '
            line = line[:17] + f"{new_resname:>3}" + line[20:]
            line = line[:12] + atom_name + line[16:]

        out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_amber.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    logger.info(f"  Normalized atom/residue names -> {out_path}")
    return out_path


def normalize_pdb_for_rosetta(pdb_path):
    """Rename atoms and residues from AMBER/MCSCE conventions back to Rosetta conventions.

    Reverses the AMBER normalization so that Rosetta can read MCSCE output PDBs.

    Operates on fixed-width PDB columns:
      - Atom name:    cols 12-16
      - Residue name: cols 17-20

    AMBER → Rosetta mappings applied:
      Residue names: HID→HIS_D, HIE→HIS, HIP→HIS (Rosetta auto-detects protonation)
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    # AMBER → Rosetta residue name mapping
    RESNAME_MAP = {
        'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    }

    out_lines = []
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            out_lines.append(line)
            continue

        resname = line[17:20].strip()

        # Residue name normalization back to Rosetta conventions
        if resname in RESNAME_MAP:
            new_resname = RESNAME_MAP[resname]
            line = line[:17] + f"{new_resname:>3}" + line[20:]

        out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_rosetta.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    logger.info(f"  Normalized AMBER -> Rosetta names -> {out_path}")
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


def renumber_residues(pdb_path):
    """Renumber residues contiguously starting from 1.

    MCSCE assumes contiguous residue numbering (no gaps) when iterating
    over residues via `idx + structure.res_nums[0]`. CATH domain PDBs
    often have non-contiguous numbering from the parent chain, which
    causes IndexError when MCSCE tries to filter for a non-existent
    residue number.

    Operates on fixed-width PDB columns:
      - Residue sequence number: cols 22-26 (right-justified)
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    # Build mapping from (chain, old_resnum) -> new_resnum
    seen = {}
    counter = 0
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        chain = line[21]
        old_resnum = line[22:26].strip()
        key = (chain, old_resnum)
        if key not in seen:
            counter += 1
            seen[key] = counter

    out_lines = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            chain = line[21]
            old_resnum = line[22:26].strip()
            new_resnum = seen[(chain, old_resnum)]
            line = line[:22] + f"{new_resnum:>4}" + line[26:]
        out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_renum.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    logger.info(f"  Renumbered {counter} residues contiguously -> {out_path}")
    return out_path


def _cleanup_temp_files(temp_files):
    """Remove a list of intermediate temp file paths."""
    for f in temp_files:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


def run_mcsce(pdb_path, outdir, n_conformers, temperature, failed_log=None,
              minimize=True, preprocess=False, seed=42):
    """Run MCSCE on a single PDB file. Based on the wrapper mcsce_sidechain.py from the MCSCE repo.

    Args:
        preprocess: If True, run idealize + repack + 2-step minimize with
            HNQ flipping on the input structure before MC-SCE.
        minimize: If True, run Rosetta cartesian minimization on each
            generated conformer.
    """
    original_pdb_path = pdb_path
    temp_files = []  # track intermediate files for cleanup

    # Step 1-2: Repack + minimize (idealize, repack, 2-stage min w/ HNQ flipping)
    if preprocess:
        pdb_path = minimize_structure(pdb_path, seed=seed)
        temp_files.append(pdb_path)

    # Step 3: Remove side chains (keep backbone + CB + H atoms per MC-SCE)
    pdb_path = remove_sidechain(pdb_path)
    temp_files.append(pdb_path)

    # Normalize Rosetta atom/residue names to AMBER conventions for MCSCE
    pdb_path = normalize_pdb_for_amber(pdb_path)
    temp_files.append(pdb_path)

    # Pre-filter: skip PDBs with non-standard residues
    is_valid, bad_residues = validate_pdb_for_mcsce(pdb_path)
    if not is_valid:
        bad_str = ", ".join(f"{r[0]}:{r[2]}{r[1]}" for r in bad_residues)
        logger.error(f"SKIPPED {Path(pdb_path).stem}: non-standard residues: {bad_str}")
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{original_pdb_path}\tNON_STANDARD_RESIDUES\t{bad_str}\n")
        _cleanup_temp_files(temp_files)
        return []

    # Renumber residues contiguously (1, 2, 3, ...) — required because
    # MCSCE's initialize_func_calc assumes contiguous numbering when it
    # computes residue indices as idx + res_nums[0]. CATH domain PDBs
    # often have gaps that cause IndexError.
    pdb_path = renumber_residues(pdb_path)
    temp_files.append(pdb_path)

    import numpy as np
    from functools import partial
    from mcsce.libs.libstructure import Structure
    from mcsce.core.side_chain_builder import initialize_func_calc, create_side_chain_ensemble
    from mcsce.core.build_definitions import forcefields
    from mcsce.core.definitions import aa3to1
    from mcsce.libs.libenergy import prepare_energy_function

    stem = Path(pdb_path).stem
    # Strip intermediate suffixes for output dir naming
    out_stem = stem.replace("_preproc", "").replace("_bb", "").replace("_amber", "").replace("_renum", "")
    out_sub = os.path.join(outdir, out_stem)
    os.makedirs(out_sub, exist_ok=True)

    logger.info(f"Running MC-SCE on {out_stem} ({n_conformers} conformers, T={temperature}K)")

    # Step 4: MC-SCE side chain ensemble generation
    # Build Structure from FASTA rather than loading from PDB directly.
    # This matches the working mcsce_sidechain.py wrapper and avoids:
    #   - KeyError from terminal residue/atom mismatches with forcefield
    #     (PDB uses HID but forcefield expects HIP for terminal histidines)
    #   - IndexError from residue_types returning list instead of numpy array
    #   - Atom naming inconsistencies between Rosetta/AMBER/MCSCE conventions
    #
    # The FASTA builder creates internally consistent atom arrays that are
    # guaranteed to match the forcefield definitions.

    # First load PDB to extract sequence and backbone coords
    pdb_structure = Structure(Path(pdb_path))
    pdb_structure.build()
    res_types = list(pdb_structure.residue_types)
    logger.info(f"  Residue sequence ({len(res_types)} res): {' '.join(res_types)}")

    # Build coord lookup from PDB: (resnum, atom_name) -> [x, y, z]
    pdb_atoms = pdb_structure.data_array
    from mcsce.libs.libstructure import col_resSeq, col_name, col_x, col_y, col_z
    coord_lookup = {}
    for row in pdb_atoms:
        key = (int(row[col_resSeq]), row[col_name])
        coord_lookup[key] = np.array([float(row[col_x]), float(row[col_y]), float(row[col_z])])

    # Convert 3-letter residue codes to 1-letter FASTA.
    # All histidine variants (HID, HIE, HIP) must map to 'H' so that
    # parse_fasta_to_array → translate_seq_to_3l converts them to 'HIP',
    # which is what the forcefield expects at terminal positions (NHIP, CHIP).
    # aa3to1 maps HID→'d', HIE→'e', HIP→'p' (lowercase), but translate_seq_to_3l
    # only handles 'H'→'HIP'. Using lowercase would keep HID/HIE and cause
    # KeyError when the forcefield looks up NHID or CHID.
    HIS_VARIANTS = {'HID', 'HIE', 'HIP', 'HIS'}
    fasta = ""
    for res in res_types:
        if res in HIS_VARIANTS:
            fasta += 'H'
        else:
            code = aa3to1.get(res, None)
            if code is None:
                logger.error(f"SKIPPED {out_stem}: unknown residue {res} has no 1-letter code")
                if failed_log:
                    with open(failed_log, "a") as fh:
                        fh.write(f"{original_pdb_path}\tUNKNOWN_RESIDUE\t{res}\n")
                _cleanup_temp_files(temp_files)
                return []
            fasta += code

    # Build MCSCE Structure from FASTA (internally consistent naming)
    structure = Structure(fasta=fasta)
    structure.build()

    # Map PDB backbone coords onto the FASTA structure's atom order.
    # FASTA structure atom order per residue:
    #   N-term non-PRO: N, CA, C, O, H1, H2, H3
    #   N-term PRO:     N, CA, C, O, H1, H2
    #   Middle non-PRO: N, CA, C, O, H
    #   Middle PRO:     N, CA, C, O
    #   C-term gets +OXT appended
    n_atoms = len(structure.data_array)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    # Renumbered PDB residues start at 1, matching FASTA resid = residx + 1
    for i, row in enumerate(structure.data_array):
        resnum = int(row[col_resSeq])
        atom_name = row[col_name]
        key = (resnum, atom_name)
        if key in coord_lookup:
            coords[i] = coord_lookup[key]
        else:
            # H atom missing from PDB — approximate from backbone geometry
            # Place H ~1.0 Å from N along the C(prev)-N direction
            n_key = (resnum, 'N')
            ca_key = (resnum, 'CA')
            if n_key in coord_lookup and ca_key in coord_lookup:
                n_pos = coord_lookup[n_key]
                ca_pos = coord_lookup[ca_key]
                # Place H opposite to CA direction from N
                direction = n_pos - ca_pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    coords[i] = n_pos + direction / norm * 1.0
                else:
                    coords[i] = n_pos
            else:
                coords[i] = [0.0, 0.0, 0.0]
            logger.debug(f"  Approximated position for {atom_name} at res {resnum}")

    structure.coords = coords

    try:
        # No need to call remove_side_chains() — the FASTA-built structure
        # already contains only backbone atoms (N, CA, C, O, H, terminals).

        # Initialize energy calculators (required before ensemble generation)
        # Use ["lj", "clash"] terms — matches the working mcsce_sidechain.py wrapper
        ff = forcefields["Amberff14SB"]
        ff_obj = ff(Cterminal='OXT', Nterminal='HN')
        initialize_func_calc(
            partial(prepare_energy_function, batch_size=16,
                    forcefield=ff_obj, terms=["lj", "clash"]),
            structure=structure,
            aa_seq=list(structure.residue_types),
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
        _cleanup_temp_files(temp_files)
        return []

    _cleanup_temp_files(temp_files)

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

    # Step 5: Rosetta cartesian coordinate minimization on each conformer
    # Normalize AMBER/MCSCE naming back to Rosetta conventions first
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
        # Convert AMBER naming -> Rosetta naming for cartesian minimization
        rosetta_pdb = normalize_pdb_for_rosetta(str(pdb_out))
        pose = pyrosetta.pose_from_pdb(rosetta_pdb)
        score_before = scorefxn_cart(pose)
        min_mover.apply(pose)
        score_after = scorefxn_cart(pose)
        # Overwrite original MCSCE output with minimized Rosetta-named structure
        pose.dump_pdb(str(pdb_out))
        logger.info(f"  Minimized {pdb_out.name}: {score_before:.1f} -> {score_after:.1f}")
        minimized_paths.append(str(pdb_out))
        # Clean up temporary rosetta-named PDB
        if os.path.exists(rosetta_pdb):
            os.remove(rosetta_pdb)

    return minimized_paths

# main functions for parsing args from shell script or command line 
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
