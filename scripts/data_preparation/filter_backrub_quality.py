#!/usr/bin/env python
"""Filter backrub conformers by Rosetta score, sequence gaps, and DRMSD.

Imports scoring/RMSD from existing modules. Criteria:
  1. Rosetta score > 0 or outside [-1200, -400] -> remove
  2. Sequence gaps in residue numbering -> remove
  3. DRMSD > 10 A vs original -> remove

Outputs filter_results.csv and statistics plots (RMSD, seq identity, chi KDE).

Usage:
    python scripts/data_preparation/filter_backrub_quality.py \
        --data-dir /path/to/dataset --output-dir /path/to/filtered
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser

# Reuse existing functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rmsd_backrub_vs_original import get_ca_atoms, compute_rmsd_kabsch, compute_drmsd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from plot_chi_angle_kde import extract_chi_angles_from_dir, plot_kde_for_amino_acid, AA_WITH_CHI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ── Rosetta scoring (same score function as run_backrub.py: beta_nov16) ───────

def init_rosetta():
    """Initialize PyRosetta once, return the score function."""
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    pyrosetta.init(
        "-ignore_unrecognized_res -mute all "
        "-ignore_zero_occupancy false "
        "-corrections:beta_nov16",
        set_logging_handler=None,
    )
    return ScoreFunctionFactory.create_score_function("beta_nov16")


def score_pdb(pdb_path, scorefxn):
    """Score a PDB with the provided Rosetta score function."""
    import pyrosetta
    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        return scorefxn(pose)
    except Exception as e:
        logger.warning("Score failed for %s: %s", pdb_path, e)
        return None


# ── Sequence gap detection ────────────────────────────────────────────────────

def has_sequence_gaps(pdb_path):
    """Check for gaps in residue numbering (missing residues)."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception:
        return True
    for model in structure:
        for chain in model:
            res_ids = [res.id[1] for res in chain if res.id[0] == " "]
            for i in range(1, len(res_ids)):
                if res_ids[i] - res_ids[i - 1] > 1:
                    return True
        break
    return False


# ── Sequence identity ─────────────────────────────────────────────────────────

def get_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    seq = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":
                    seq.append(THREE_TO_ONE.get(res.get_resname().strip(), "X"))
        break
    return "".join(seq)


def sequence_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    return sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len])) / min_len


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter backrub dataset by score, gaps, DRMSD")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root dir with 'originals/' and ensemble subdirs")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                        help="Ensemble dir (default: <data-dir>/ai-cath_backrub_subset_ensembles)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dir for filtered conformers and results")
    parser.add_argument("--score-min", type=float, default=-1200)
    parser.add_argument("--score-max", type=float, default=-400)
    parser.add_argument("--drmsd-max", type=float, default=10.0)
    parser.add_argument("--skip-chi", action="store_true", help="Skip chi angle KDE plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    orig_dir = data_dir / "originals"
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else data_dir / "ai-cath_backrub_subset_ensembles"
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for d, name in [(orig_dir, "originals"), (ensemble_dir, "ensembles")]:
        if not d.is_dir():
            logger.error("%s dir not found: %s", name, d)
            sys.exit(1)

    scorefxn = init_rosetta()
    orig_pdbs = sorted(orig_dir.glob("*.pdb"))
    logger.info("Found %d original PDBs", len(orig_pdbs))

    records = []
    for orig_pdb in orig_pdbs:
        stem = orig_pdb.stem
        conf_dir = ensemble_dir / stem
        if not conf_dir.is_dir():
            continue

        _, ref_coords = get_ca_atoms(orig_pdb)
        orig_seq = get_sequence(orig_pdb)
        if len(ref_coords) < 3:
            continue

        for conf_pdb in sorted(conf_dir.glob("*.pdb")):
            rec = {"protein": stem, "conformer": conf_pdb.name, "kept": False,
                   "removal_reason": None, "rosetta_score": None,
                   "drmsd": None, "rmsd": None, "seq_identity": None, "n_residues": None}

            if has_sequence_gaps(conf_pdb):
                rec["removal_reason"] = "sequence_gap"
                records.append(rec)
                continue

            score = score_pdb(conf_pdb, scorefxn)
            rec["rosetta_score"] = score
            if score is None or score > 0 or score < args.score_min or score > args.score_max:
                rec["removal_reason"] = "score"
                records.append(rec)
                continue

            _, mob_coords = get_ca_atoms(conf_pdb)
            drmsd = compute_drmsd(ref_coords, mob_coords)
            rec["drmsd"] = drmsd
            rec["rmsd"] = compute_rmsd_kabsch(ref_coords, mob_coords)
            rec["n_residues"] = min(len(ref_coords), len(mob_coords))

            if np.isnan(drmsd) or drmsd > args.drmsd_max:
                rec["removal_reason"] = "drmsd"
                records.append(rec)
                continue

            rec["seq_identity"] = sequence_identity(orig_seq, get_sequence(conf_pdb))
            rec["kept"] = True
            records.append(rec)

            # Copy kept conformer
            dst = output_dir / stem
            dst.mkdir(exist_ok=True)
            shutil.copy2(conf_pdb, dst / conf_pdb.name)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "filter_results.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(df)
    kept_df = df[df["kept"]]
    print("\n" + "=" * 70)
    print("BACKRUB QUALITY FILTER SUMMARY")
    print("=" * 70)
    print(f"  Conformers: {kept_df.shape[0]}/{total} kept ({kept_df.shape[0]/max(total,1)*100:.1f}%)")
    print(f"  Proteins with >= 1 kept: {kept_df['protein'].nunique()}/{df['protein'].nunique()}")
    removed = df[~df["kept"]]
    if len(removed):
        print(f"  Removal reasons:")
        for reason, n in removed["removal_reason"].value_counts().items():
            print(f"    {reason}: {n}")
    if len(kept_df):
        for col, label in [("rosetta_score", "Score"), ("rmsd", "RMSD"), ("drmsd", "DRMSD"), ("seq_identity", "SeqID")]:
            vals = kept_df[col].dropna()
            if len(vals):
                print(f"  {label}: {vals.mean():.3f} +/- {vals.std():.3f}")
    print("=" * 70 + "\n")

    # ── Plots (seaborn histplot + KDE, matching cath20_analysis.ipynb) ────
    if len(kept_df) == 0:
        return

    for col, label, color in [("rmsd", "Ca RMSD (A)", "#2a9d8f"),
                               ("drmsd", "Ca DRMSD (A)", "coral"),
                               ("rosetta_score", "Rosetta Score (REU)", "#e76f51"),
                               ("seq_identity", "Sequence Identity", "#264653")]:
        vals = kept_df[col].dropna()
        if len(vals) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(vals, bins=30, color=color, edgecolor="white", linewidth=0.3, kde=True, ax=ax)
        ax.axvline(vals.mean(), color=color, linestyle="--", alpha=0.7,
                   label=f"mean={vals.mean():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} distribution (n={len(vals)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"filtered_{col}_hist.png", dpi=150)
        plt.close(fig)

    # Chi angle KDE plots (reuse plot_chi_angle_kde.py functions)
    if not args.skip_chi:
        logger.info("Generating chi angle KDE plots...")
        chi_data = {"Filtered": extract_chi_angles_from_dir(str(output_dir))}
        for resname in AA_WITH_CHI:
            out_path = str(figures_dir / f"chi_angles_{resname}.png")
            plot_kde_for_amino_acid(resname, chi_data, out_path)

    logger.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()
