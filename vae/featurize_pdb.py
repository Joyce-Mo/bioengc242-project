"""Featurize PDB files into (C, L, L) numpy stacks for vae.py.

Reads a PDB (or a list / directory of PDBs) and outputs one .npy file per
input. The output is the 7-channel feature tensor that `ProteinFeatureDataset`
in vae/vae.py loads at training time:

Full channel list (copied over for vae.py lol)
    ch 0  : Calpha-Calpha distance map      (angstrom, /D_MAX, clipped to [0, 1])
    ch 1  : Cbeta-Cbeta distance map        (angstrom, /D_MAX, clipped to [0, 1])
                                              (GLY uses CA in place of CB)
    ch 2  : binary contact map              (Calpha-Calpha < 8 angstrom)
    ch 3  : hydrophobicity outer product    (Kyte-Doolittle, scaled to [0, 1])
    ch 4  : charge outer product            (-1 / 0 / +1, shifted to [0, 1])
    ch 5  : polarity outer product          (binary: polar vs nonpolar)
    ch 6  : per-residue SASA outer product  (angstrom^2, /SASA_MAX, clipped)

Per-residue scalars f_i are turned into LxL channels via the outer product
0.5 * (f_i + f_j) so the feature stack is symmetric like the distance maps. 

Based on AlphaFold "pair feature" convention and lets the 2D conv stack 
in vae.py see the pairs as a single image.

D_MAX, SASA_MAX, and N_CHANNELS are copied and pasted over....
but TODO might refactor these or if I want to make this more general-purpose later. 
maybe will update this to take in yaml config or parse more args. 

Usage: python vae/featurize_pdb.py --pdb path/to/pdb --outdir features/
"""

from __future__ import print_function
import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


# Constants
#
# Must stay in sync with vae/vae.py ; duplicated rather than imported so
# that featurization doesn't fire the top-level argparse in vae.py.
D_MAX = 22.0          # angstrom ; distance map clip / normalization
SASA_MAX = 200.0      # angstrom^2 ; per-residue SASA normalization
N_CHANNELS = 7
CONTACT_THRESHOLD = 8.0  # angstrom ; Calpha-Calpha cutoff for the contact map


# # Per-residue scalar features
#
# Standard tables. Hydrophobicity from Kyte-Doolittle (1982); charge from
# physiological pH conventions (K, R = +1; D, E = -1; H ~ 0); polarity from
# the standard polar / nonpolar partition (H, K, R, D, E, S, T, N, Q, Y, C
# polar; the rest nonpolar). Glycine is treated as nonpolar by convention.

KD_HYDROPHOBICITY = {
    'ALA':  1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS':  2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE':  4.5,
    'LEU':  3.8, 'LYS': -3.9, 'MET':  1.9, 'PHE':  2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL':  4.2,
}
# Kyte-Doolittle ranges from -4.5 (ARG) to +4.5 (ILE). Rescale to [0, 1].
_KD_MIN, _KD_MAX = -4.5, 4.5

CHARGE = {
    'ARG':  1.0, 'LYS':  1.0,
    'ASP': -1.0, 'GLU': -1.0,
}  # all others default to 0

POLAR_RESIDUES = {
    'ARG', 'LYS', 'ASP', 'GLU', 'HIS', 'SER', 'THR',
    'ASN', 'GLN', 'TYR', 'CYS',
}

# Treat AMBER protonation-state variants of histidine as HIS for table
# lookups. (vae.py and run_mcsce.py both deal with HID/HIE/HIP.)
HIS_VARIANTS = {'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'}


def _canon_resname(resname):
    if resname in HIS_VARIANTS:
        return 'HIS'
    return resname


def _residue_scalars(resname):
    """Return (hydrophobicity, charge, polarity) all scaled to [0, 1]."""
    rn = _canon_resname(resname)
    kd_raw = KD_HYDROPHOBICITY.get(rn, 0.0)
    kd = (kd_raw - _KD_MIN) / (_KD_MAX - _KD_MIN)  # -> [0, 1]
    chg_raw = CHARGE.get(rn, 0.0)
    chg = (chg_raw + 1.0) / 2.0                     # -> {0.0, 0.5, 1.0}
    pol = 1.0 if rn in POLAR_RESIDUES else 0.0      # -> {0.0, 1.0}
    return kd, chg, pol


# Geometry helpers

def _get_ca_cb_coords(residue):
    """Return (ca, cb) coordinates for one residue. GLY (no CB) gets CA in
    both slots. Returns (None, None) if CA is missing."""
    if 'CA' not in residue:
        return None, None
    ca = np.asarray(residue['CA'].get_coord(), dtype=np.float32)
    if 'CB' in residue:
        cb = np.asarray(residue['CB'].get_coord(), dtype=np.float32)
    else:
        cb = ca
    return ca, cb


def _pair_distance_map(coords):
    """L x L euclidean distance matrix from an (L, 3) array of coords."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def _outer_average(vec):
    """Symmetric L x L pair feature: 0.5 * (f_i + f_j)."""
    vec = np.asarray(vec, dtype=np.float32)
    return 0.5 * (vec[:, None] + vec[None, :])


# Main featurization

def featurize_pdb(pdb_path):
    """Compute the (N_CHANNELS, L, L) feature stack for one PDB file.

    Only the first model and first chain are used (CATH domains in the
    AI-CATH subset are single-chain). Note this AI-CATH dataset is the one 
    from prtopardelle c. https://zenodo.org/records/15881564 
    
    """
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(Path(pdb_path).stem, str(pdb_path))

    # Compute SASA on the parsed structure (Shrake-Rupley); attaches a .sasa
    # attribute to every atom and residue.
    ShrakeRupley().compute(structure, level="R")

    # First model, first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())

    ca_coords, cb_coords, resnames, sasa_per_res = [], [], [], []
    n_dropped = 0
    for residue in chain:
        # skip hetero residues (waters, ligands) ; hetflag is non-blank
        if residue.id[0].strip() != "":
            continue
        ca, cb = _get_ca_cb_coords(residue)
        if ca is None:
            n_dropped += 1
            continue
        ca_coords.append(ca)
        cb_coords.append(cb)
        resnames.append(residue.get_resname().strip().upper())
        sasa_per_res.append(float(getattr(residue, "sasa", 0.0)))

    # drop ones with no CA, in case mcsce or bacrub conformations end up being weird
    # for now they seem fine, but some of the mcsce runs are not working and might cause 
    # energetically-unfavorable conformations that could have missing atoms? TBD.
    if n_dropped:
        print(f"  warning: dropped {n_dropped} residues with no CA in {pdb_path}",
              file=sys.stderr)

    L = len(ca_coords)
    if L == 0:
        raise ValueError(f"no usable residues in {pdb_path}")
    ca_coords = np.stack(ca_coords)
    cb_coords = np.stack(cb_coords)

    # ch 0, 1: distance maps (clipped + normalized to [0, 1])
    ca_dist = _pair_distance_map(ca_coords)
    cb_dist = _pair_distance_map(cb_coords)
    ca_dist_norm = np.clip(ca_dist / D_MAX, 0.0, 1.0)
    cb_dist_norm = np.clip(cb_dist / D_MAX, 0.0, 1.0)

    # ch 2: binary contact map (Calpha-Calpha < CONTACT_THRESHOLD), zero
    # diagonal so the model isn't told every residue is in contact with itself
    contact = (ca_dist < CONTACT_THRESHOLD).astype(np.float32)
    np.fill_diagonal(contact, 0.0)

    # ch 3-5: per-residue scalar == outer-average pair features
    kd, chg, pol = zip(*(_residue_scalars(r) for r in resnames))
    hydro_map = _outer_average(kd)
    charge_map = _outer_average(chg)
    polar_map = _outer_average(pol)

    # ch 6: SASA outer-average pair feature
    sasa_norm = np.clip(np.asarray(sasa_per_res, dtype=np.float32) / SASA_MAX, 0.0, 1.0)
    sasa_map = _outer_average(sasa_norm)

    feats = np.stack(
        [ca_dist_norm, cb_dist_norm, contact, hydro_map, charge_map, polar_map, sasa_map],
        axis=0,
    ).astype(np.float32)
    assert feats.shape == (N_CHANNELS, L, L), feats.shape
    return feats


# CLI parsing 

def _gather_pdb_paths(args):
    """Resolve the list of PDB paths to process from --pdb / --pdb-dir /
    --pdb-list, then optionally slice into a chunk for SLURM array jobs
    via --task-id and --n-tasks.

    Chunking: when --task-id T and --n-tasks N are both given, the full
    path list is split into N roughly-equal chunks and only chunk T is
    returned (1-indexed to match SLURM $SLURM_ARRAY_TASK_ID + 1).
    If only --task-id is given without --n-tasks, it falls back to
    selecting a single path (backwards-compatible with earlier usage).
    """
    # Step 1: resolve all paths from the chosen input mode
    if args.pdb:
        paths = [args.pdb]
    elif args.pdb_dir:
        paths = sorted(str(p) for p in Path(args.pdb_dir).rglob("*.pdb"))
        if not paths:
            raise SystemExit(f"no .pdb files found under {args.pdb_dir}")
    elif args.pdb_list:
        with open(args.pdb_list) as fh:
            paths = [line.strip() for line in fh if line.strip()]
        if not paths:
            raise SystemExit(f"no paths in {args.pdb_list}")
    else:
        raise SystemExit("provide one of --pdb, --pdb-dir, or --pdb-list")

    # Step 2: apply SLURM array chunking if requested
    if args.task_id is not None:
        if args.n_tasks is not None:
            # Chunked mode: split paths into n_tasks chunks, return chunk task_id
            chunk_size = math.ceil(len(paths) / args.n_tasks)
            start = (args.task_id - 1) * chunk_size
            end = min(start + chunk_size, len(paths))
            if start >= len(paths):
                print(f"task_id {args.task_id} exceeds path count "
                      f"({len(paths)} PDBs / {args.n_tasks} tasks); nothing to do")
                return []
            paths = paths[start:end]
        else:
            # Legacy single-path mode (backwards compat)
            idx = args.task_id - 1
            if idx < 0 or idx >= len(paths):
                raise SystemExit(
                    f"task_id {args.task_id} out of range (1-{len(paths)})")
            paths = [paths[idx]]

    return paths


def main():
    parser = argparse.ArgumentParser(description="Featurize PDB(s) into (C, L, L) .npy stacks for vae.py")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", type=str, help="single PDB file")
    src.add_argument("--pdb-dir", type=str, help="directory of PDBs (recursively globbed)")
    src.add_argument("--pdb-list", type=str, help="text file with one PDB path per line")
    parser.add_argument("--task-id", type=int, default=None,
                        help="1-indexed SLURM array task id; selects a chunk of "
                             "PDBs to process (works with any input mode)")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="total number of SLURM array tasks; splits the PDB "
                             "list into --n-tasks equal chunks (requires --task-id)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to write {stem}.npy files into")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-featurize PDBs whose .npy already exists")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdb_paths = _gather_pdb_paths(args)
    print(f"featurizing {len(pdb_paths)} PDB(s) -> {outdir}")

    n_ok, n_skip, n_fail, n_empty = 0, 0, 0, 0
    for pdb_path in pdb_paths:
        out_path = outdir / f"{Path(pdb_path).stem}.npy"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        # Quick check: skip 0-byte files without invoking BioPython
        if os.path.getsize(pdb_path) == 0:
            n_empty += 1
            print(f"  SKIP {pdb_path}: 0-byte file", file=sys.stderr)
            continue
        try:
            feats = featurize_pdb(pdb_path)
        except Exception as e:
            print(f"  FAIL {pdb_path}: {type(e).__name__}: {e}", file=sys.stderr)
            n_fail += 1
            continue
        np.save(out_path, feats)
        n_ok += 1
        print(f"  wrote {out_path}  shape={feats.shape}")

    print(f"done: {n_ok} written, {n_skip} skipped (exist), "
          f"{n_empty} skipped (empty), {n_fail} failed")


if __name__ == "__main__":
    main()
