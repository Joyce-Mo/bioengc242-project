#!/usr/bin/env python
"""Run MC-SCE ensemble generation on a batch of PDB files.
Regular PDB files iwth side chains (not minimzed) can be used as input: 
  - PDBData strips side chains and rebuilds them from AMBER templates
  - the arg --fix uses PDBFixer to add missing atoms/hydrogens, replace
    nonstandard residues, and remove heterogens
  - --relax (optional) does an OpenMM backbone relaxation in solvent
  - resolve_clash() runs an OpenMM energy minimization with the
    backbone fixed on each generated conformer

Environment setup:
This script does in-process imports from mcsce-precompute. The Python
env you run this from must therefore have ALL of the following installed:

  mcsce-precompute deps:
    - jax, jaxlib      (pip install "jax[cpu]"  — or "jax[cuda12]" for GPU)
    - scipy
    - numpy
    - openmm           (conda install -c conda-forge openmm)
    - pdbfixer         (conda install -c conda-forge pdbfixer)
    - biopython        (pip install biopython)
    - tqdm             (pip install tqdm)

  + this wrapper's deps for the post-MCSCE cartesian min step:
    - pyrosetta        (https://www.pyrosetta.org/downloads)

Override the mcsce-precompute checkout location per host by exporting
MCSCE_PRECOMPUTE_DIR before launching this script.

Usage:
    python run_mcsce.py --pdb input.pdb --outdir ensembles/mcsce --nconfs 5

Designed for HPC job arrays. Each job processes one PDB.
"""

import argparse
import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

# Set multiprocessing start method to "spawn" BEFORE jax (or anything that
# imports it) gets loaded. mcsce-precompute uses multiprocessing.Pool
# internally; on Linux Pool defaults to fork, but JAX is multithreaded and
# fork-after-thread leads to deadlock — see the runtime warning:
#   "os.fork() ... JAX is multithreaded, so this will likely lead to a
#    deadlock."
# mcsce-precompute does this in its own __main__, but we import it as a
# module so that block never runs.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Verify spawn is active — fork + JAX threads causes deadlocks.
if multiprocessing.get_start_method() != "spawn":
    logger.warning(
        f"multiprocessing start method is '{multiprocessing.get_start_method()}', "
        "not 'spawn'. JAX deadlocks are possible."
    )

# Force unbuffered output — ensures HPC logs are written before crashes
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Path to local mcsce-precompute checkout. Override per-host by exporting
# MCSCE_PRECOMPUTE_DIR before launching this script (e.g. in your sbatch /
# qsub wrapper). Otherwise the local-laptop default below is used.
#
# The directory uses bare imports (e.g. `from protein_data import PDBData`),
# so we have to add it to sys.path before importing ensemble_gen.
MCSCE_PRECOMPUTE_DIR = os.environ.get(
    "MCSCE_PRECOMPUTE_DIR",
    "/wynton/home/rotation/jqmo/rotation3/mcsce-precompute"
    # "/Users/joycemo/Documents/GitHub/mcsce-precompute",
)
if MCSCE_PRECOMPUTE_DIR not in sys.path:
    sys.path.insert(0, MCSCE_PRECOMPUTE_DIR)


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


def normalize_pdb_for_rosetta(pdb_path):
    """Rename residues from AMBER conventions back to Rosetta conventions.

    mcsce-precompute outputs AMBER-named PDBs (HID/HIE/HIP). Rosetta needs
    HIS — it auto-detects protonation state.

    Operates on fixed-width PDB columns:
      - Residue name: cols 17-20
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    RESNAME_MAP = {'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS'}

    out_lines = []
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            out_lines.append(line)
            continue
        resname = line[17:20].strip()
        if resname in RESNAME_MAP:
            line = line[:17] + f"{RESNAME_MAP[resname]:>3}" + line[20:]
        out_lines.append(line)

    out_path = str(Path(pdb_path).parent / f"{Path(pdb_path).stem}_rosetta.pdb")
    with open(out_path, 'w') as f:
        f.writelines(out_lines)
    return out_path


def normalize_pdb_for_mcsce(pdb_path, out_path):
    """Rename AMBER protonation/disulfide variants to canonical 3-letter codes
    that mcsce-precompute's PDBData understands.

    PDBFixer leaves CYX (disulfide-bonded Cys) and HID/HIE/HIP (His protonation
    variants) alone because they're "standard" to AMBER, but mcsce-precompute's
    PDBData.prepare_sidechain does a dict lookup on residue name and only
    knows the canonical 3-letter codes (CYS, HIS), causing
    `KeyError: 'CYX'` etc.

    PDBData handles disulfides geometrically via get_ss_bonds (distance-based),
    so renaming CYX → CYS is safe — the SS bond is detected from coords.
    """
    RESNAME_MAP = {
        'CYX': 'CYS', 'CYM': 'CYS',
        'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
        'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS',
    }
    with open(pdb_path) as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if not line.startswith(('ATOM', 'HETATM')):
            out_lines.append(line)
            continue
        resname = line[17:20].strip()
        if resname in RESNAME_MAP:
            line = line[:17] + f"{RESNAME_MAP[resname]:>3}" + line[20:]
        out_lines.append(line)
    with open(out_path, 'w') as f:
        f.writelines(out_lines)
    return out_path


def run_mcsce(pdb_path, outdir, n_conformers, temperature, failed_log=None,
              minimize=True, fix=True, relax=False, device="CUDA",
              energy_funcs=("vdw", "elec", "hpmf"), keep_disulfide=True,
              clean=True, num_workers=1, seed=42, keep_workdir=False):
    """Run mcsce-precompute on a single PDB file.

    Args:
        pdb_path: input PDB (regular PDB with side chains; missing atoms OK if fix=True).
        outdir:   directory to write final conformer ensemble into.
        
        n_conformers: number of regrown conformer attempts (passed as `num_attempts`).
        temperature: kept for CLI compatibility — mcsce-precompute uses
            separate `growing_kT` / `rejection_kT` knobs internally; this
            argument is currently unused but retained so existing job
            scripts don't break.
        
        fix:      run PDBFixer (add hydrogens, missing atoms, replace
            nonstandard residues). Recommended ON for raw PDBs.
        
        relax:    run OpenMM solvent backbone relaxation before MCSCE.
            Default OFF; only enable if your input backbone is bad.
        
        minimize: run Rosetta cartesian minimization on each generated
            conformer (Step 5).
        
        device:   "CUDA" or "CPU" — passed through to mcsce-precompute's
            OpenMM platform selection.
        
        keep_workdir: if True, keep the {outdir}/{stem}/_mcsce_work cache
            (rotamer/energy pickles) instead of deleting it after success.
    """
    try:
        from prepare_protein import fix_protein, relax_backbone
        from ensemble_gen import ensemble_gen
        from definitions import RES_RADII
    except ImportError as e:
        logger.error(
            f"Failed to import mcsce-precompute modules: {e}\n"
            f"  Make sure MCSCE_PRECOMPUTE_DIR points at your local clone "
            f"({MCSCE_PRECOMPUTE_DIR}) and that ALL of "
            f"jax/scipy/numpy/openmm/pdbfixer/biopython/tqdm "
            f"are installed in this Python env."
        )
        raise

    # mcsce-precompute creates CYX residues for disulfide cysteines
    # (protein_data.py:315-316) but RES_RADII in definitions.py is
    # missing the CYX entry, causing KeyError in rotamer_important_pairs.
    # Patch the shared dict object so all modules see the fix.
    if 'CYX' not in RES_RADII:
        RES_RADII['CYX'] = RES_RADII['CYS']

    original_pdb_path = pdb_path
    stem = Path(pdb_path).stem

    out_sub = os.path.join(outdir, stem)
    os.makedirs(out_sub, exist_ok=True)
    work_dir = os.path.join(out_sub, "_mcsce_work")
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"Running mcsce-precompute on {stem} "
                f"({n_conformers} attempts, device={device})")

    # Step 1: PDBFixer (add hydrogens, replace nonstandard residues)
    if fix:
        fixed_path = os.path.join(work_dir, f"{stem}_fixed.pdb")
        try:
            fix_protein(pdb_path, fixed_path)
        except Exception as e:
            logger.error(f"PDBFixer failed on {stem}: {type(e).__name__}: {e}")
            if failed_log:
                with open(failed_log, "a") as fh:
                    fh.write(f"{original_pdb_path}\tPDBFIXER_FAIL\t{e}\n")
            return []
        pdb_path = fixed_path

    # Step 2 (optional): OpenMM solvent backbone relaxation
    if relax:
        relaxed_path = os.path.join(work_dir, f"{stem}_relaxed.pdb")
        try:
            # relax_backbone calls fix_protein internally if fix=True; we
            # already fixed above so disable it here.
            relax_backbone(pdb_path, output_path_relaxed=relaxed_path,
                           device=device, fix=False)
        except Exception as e:
            logger.error(f"Backbone relax failed on {stem}: {type(e).__name__}: {e}")
            if failed_log:
                with open(failed_log, "a") as fh:
                    fh.write(f"{original_pdb_path}\tRELAX_FAIL\t{e}\n")
            return []
        pdb_path = relaxed_path

    # Step 2b: Rename AMBER variants (CYX, HID/HIE/HIP, etc.) to canonical
    # 3-letter codes that PDBData understands. PDBFixer leaves these alone.
    normalized_path = os.path.join(work_dir, f"{stem}_normalized.pdb")
    normalize_pdb_for_mcsce(pdb_path, normalized_path)
    pdb_path = normalized_path

    # Remove stale pickle caches from previous failed runs so that
    # recomputation uses the current (normalized) PDB.
    for _pkl in Path(work_dir).glob("*.pkl"):
        logger.info(f"Removing stale cache: {_pkl.name}")
        _pkl.unlink()

    # Step 3: mcsce-precompute side chain ensemble generation.
    # Outputs land in {work_dir}/regrown_structures/regrow_*.pdb
    try:
        ensemble_gen(
            protein_path=pdb_path,
            num_attempts=n_conformers,
            temp_folder=work_dir,
            energy_funcs=list(energy_funcs),
            keep_disulfide=keep_disulfide,
            clean=clean,
            device=device,
            num_workers=num_workers,
        )
    except Exception as e:
        logger.error(f"mcsce-precompute failed on {stem}: {type(e).__name__}: {e}")
        if failed_log:
            with open(failed_log, "a") as fh:
                fh.write(f"{original_pdb_path}\t{type(e).__name__}\t{e}\n")
        return []

    regrown_dir = Path(work_dir) / "regrown_structures"
    regrown = sorted(regrown_dir.glob("regrow_*.pdb")) if regrown_dir.exists() else []

    if len(regrown) == 0:
        if device == "CPU":
            logger.info(
                f"Precomputation only (device=CPU) for {stem} — "
                f"rerun with --device CUDA to generate conformers"
            )
        else:
            logger.warning(f"Generated 0 conformers for {stem}")
            if failed_log:
                with open(failed_log, "a") as fh:
                    fh.write(f"{original_pdb_path}\tZERO_CONFORMERS\t\n")
        return []

    # Move regrown conformers up into out_sub with cleaner names
    final_paths = []
    for i, src in enumerate(regrown):
        dst = os.path.join(out_sub, f"{stem}_conf{i}.pdb")
        shutil.copyfile(src, dst)
        final_paths.append(dst)
    logger.info(f"Generated {len(final_paths)} conformers -> {out_sub}")

    if not minimize:
        if not keep_workdir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return final_paths

    # Step 4: Rosetta cartesian minimization on each conformer.
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap

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
    for conf_path in final_paths:
        rosetta_pdb = normalize_pdb_for_rosetta(conf_path)
        try:
            pose = pyrosetta.pose_from_pdb(rosetta_pdb)
            score_before = scorefxn_cart(pose)
            min_mover.apply(pose)
            score_after = scorefxn_cart(pose)
            pose.dump_pdb(conf_path)
            logger.info(f"  Minimized {Path(conf_path).name}: {score_before:.1f} -> {score_after:.1f}")
            minimized_paths.append(conf_path)
        except Exception as e:
            logger.error(f"  Cartesian min failed on {Path(conf_path).name}: {type(e).__name__}: {e}")
        finally:
            if os.path.exists(rosetta_pdb):
                os.remove(rosetta_pdb)

    if not keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return minimized_paths


def main():
    parser = argparse.ArgumentParser(description="MC-SCE ensemble generation (mcsce-precompute backend)")
    parser.add_argument("--pdb", type=str, help="Single PDB file path")
    parser.add_argument("--pdb_list", type=str, help="Text file with one PDB path per line")
    parser.add_argument("--task_id", type=int, default=None,
                        help="1-indexed task ID (from $SGE_TASK_ID or $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--nconfs", type=int, default=5,
                        help="Number of conformer attempts (default: 5)")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="(unused, kept for CLI compatibility)")
    parser.add_argument("--failed_log", type=str, default=None,
                        help="File to append failed PDB paths to")
    parser.add_argument("--no_minimize", action="store_true",
                        help="Skip Rosetta cartesian minimization after MC-SCE")
    parser.add_argument("--no_fix", action="store_true",
                        help="Skip PDBFixer preprocessing (only use if input is already fixed/protonated)")
    parser.add_argument("--relax", action="store_true",
                        help="Run OpenMM solvent backbone relaxation before MCSCE")
    parser.add_argument("--device", type=str, default="CUDA", choices=["CUDA", "CPU"],
                        help="OpenMM platform for energy steps (default: CUDA)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers for ensemble generation (default: 1)")
    parser.add_argument("--energy_funcs", type=str, default="vdw,elec,hpmf",
                        help="Comma-separated mcsce energy terms (default: vdw,elec,hpmf)")
    parser.add_argument("--no_keep_disulfide", action="store_true",
                        help="Disable disulfide bond preservation")
    parser.add_argument("--keep_workdir", action="store_true",
                        help="Keep the per-PDB _mcsce_work cache directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Rosetta (default: 42)")
    args = parser.parse_args()

    pdb_path = get_pdb_path(args)
    if not os.path.isfile(pdb_path):
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_mcsce(
        pdb_path, args.outdir, args.nconfs, args.temperature,
        failed_log=args.failed_log,
        minimize=not args.no_minimize,
        fix=not args.no_fix,
        relax=args.relax,
        device=args.device,
        energy_funcs=tuple(args.energy_funcs.split(",")),
        keep_disulfide=not args.no_keep_disulfide,
        num_workers=args.num_workers,
        seed=args.seed,
        keep_workdir=args.keep_workdir,
    )


if __name__ == "__main__":
    main()
