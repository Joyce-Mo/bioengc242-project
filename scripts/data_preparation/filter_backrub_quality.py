#!/usr/bin/env python
"""Filter PDB datasets by minimized Rosetta score, sequence gaps, repeat residues, and RMSD.

Two modes:
  --pdb-list FILE   Filter a flat list of PDBs (minimize+score, gap, repeat, RMSD checks).
                    Writes kept paths to <output-dir>/kept_pdb_list.txt.
  --data-dir DIR    Filter backrub ensembles vs originals (minimize+score, gap, DRMSD).

Criteria:
  1. Consecutive repeat residues >= 5 of the same amino acid in a row -> remove
  2. Sequence gaps in residue numbering -> remove
  3. Rosetta score after minimization > 0 -> remove (lower energy is always better).
     Minimization protocol follows run_backrub.py: idealize, repack, flip_HNQ,
     L-BFGS min chi, L-BFGS min chi+bb (Smith & Kortemme 2008).
  4. Kabsch RMSD > rmsd-max (default 3.0 A) vs original -> remove (pdb-list mode)
     DRMSD > drmsd-max vs original -> remove (ensemble mode)

Note that the original ingraham cath dataset is in:
/wynton/scratch/jqmo/rotation_datasets/OG_ingraham_cath/dompdb

Outputs filter_results.csv and before/after comparison plots (including RMSD).

Usage:
    # Filter a flat PDB list (pre-backrub dataset validation)
    python scripts/data_preparation/filter_backrub_quality.py \
        --pdb-list ai-cath_training_pdb.txt --output-dir /path/to/filtered \
        --originals-dir /path/to/OG_ingraham_cath/dompdb

    # Filter backrub ensembles
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
from run_backrub import preprocess_pose

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


# Rosetta scoring (same score function as run_backrub.py: beta_nov16) 

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


def minimize_and_score_pdb(pdb_path, scorefxn):
    """Minimize a PDB using the shared preprocessing from run_backrub.py, then score.

    Calls preprocess_pose() which applies: idealize, repack, flip_HNQ,
    L-BFGS min chi, L-BFGS min chi+bb (Smith & Kortemme 2008/2010).

    Returns the minimized Rosetta energy score (REU), or None on failure.
    """
    import pyrosetta

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
    except Exception as e:
        logger.warning("Load failed for %s: %s", pdb_path, e)
        return None

    try:
        return preprocess_pose(pose, scorefxn)
    except Exception as e:
        logger.warning("Minimize+score failed for %s: %s", pdb_path, e)
        return None


# Sequence gap detection 

def max_sequence_gap(pdb_path):
    """Return the largest gap in residue numbering, or 0 if contiguous.

    A gap of size N means N-1 residues are missing between two consecutive
    residue IDs. Returns -1 if the structure cannot be parsed.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception:
        return -1
    largest = 0
    for model in structure:
        for chain in model:
            res_ids = [res.id[1] for res in chain if res.id[0] == " "]
            for i in range(1, len(res_ids)):
                gap = res_ids[i] - res_ids[i - 1]
                if gap > largest:
                    largest = gap
        break
    return largest


# Sequence identity 

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


def longest_repeat_run(seq):
    """Return the length of the longest run of consecutive identical residues.

    For example, 'AAALLEEE' has longest run 3 (AAA or EEE).
    Returns 0 for empty sequences, 1 for sequences with no repeats.
    """
    if len(seq) == 0:
        return 0
    max_run = 1
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
            if count > max_run:
                max_run = count
        else:
            count = 1
    return max_run


def aa_fractions(seq):
    """Return fraction of alanine and glutamate in the sequence."""
    n = len(seq)
    if n == 0:
        return 0.0, 0.0
    return seq.count("A") / n, seq.count("E") / n


def count_residues(pdb_path):
    """Count standard amino acid residues in first model."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception:
        return 0
    n = 0
    for model in structure:
        for chain in model:
            n += sum(1 for res in chain if res.id[0] == " ")
        break
    return n


def protein_stem(pdb_name):
    """Extract protein ID from conformer filename: '1jvbA02_3.pdb' -> '1jvbA02'."""
    stem = Path(pdb_name).stem
    # Strip trailing _N (conformer index)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def compute_intra_protein_drmsd(group_paths):
    """Compute max pairwise DRMSD among conformers of the same protein.

    Returns dict mapping each pdb path -> max DRMSD to any sibling conformer.
    """
    coords_cache = {}
    for p in group_paths:
        _, coords = get_ca_atoms(p)
        if len(coords) >= 3:
            coords_cache[str(p)] = coords

    paths = list(coords_cache.keys())
    max_drmsd = {p: 0.0 for p in paths}

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = compute_drmsd(coords_cache[paths[i]], coords_cache[paths[j]])
            if not np.isnan(d):
                max_drmsd[paths[i]] = max(max_drmsd[paths[i]], d)
                max_drmsd[paths[j]] = max(max_drmsd[paths[j]], d)

    return max_drmsd


# ── Presentation-ready plot style ─────────────────────────────────────────────

PLOT_RC = {
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}


# ── Before/after comparison plots ─────────────────────────────────────────────

def plot_before_after(df, col, label, color_before, color_after, figures_dir):
    """Overlaid before/after histograms for a given metric."""
    before = df[col].dropna()
    after = df[df["kept"]][col].dropna()
    if len(before) < 2:
        return
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.histplot(before, bins=40, color=color_before, edgecolor="white",
                     linewidth=0.3, kde=True, ax=ax, label=f"Before (n={len(before)})", alpha=0.5)
        if len(after) >= 2:
            sns.histplot(after, bins=40, color=color_after, edgecolor="white",
                         linewidth=0.3, kde=True, ax=ax, label=f"After (n={len(after)})", alpha=0.7)
        ax.axvline(before.mean(), color=color_before, linestyle="--", alpha=0.6)
        if len(after) >= 2:
            ax.axvline(after.mean(), color=color_after, linestyle="--", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} — Before vs After Filtering")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"before_after_{col}.png", dpi=200)
        plt.close(fig)
    logger.info("Saved %s", figures_dir / f"before_after_{col}.png")


# ── PDB-list mode (flat list, no ensemble comparison) ─────────────────────────

def run_pdb_list_mode(pdb_list_path, output_dir, figures_dir,
                      rmsd_max, originals_dir, skip_chi):
    """Filter a flat PDB list by minimized Rosetta score (>0 removed), gaps, repeats, and RMSD."""
    with open(pdb_list_path) as f:
        pdb_paths = [Path(line.strip()) for line in f if line.strip()]
    logger.info("Loaded %d PDB paths from %s", len(pdb_paths), pdb_list_path)

    # Cache original Ca coords (protein_stem -> coords array)
    orig_coords_cache = {}
    if originals_dir is not None:
        originals_dir = Path(originals_dir)
        logger.info("Originals dir: %s", originals_dir)

    scorefxn = init_rosetta()
    records = []

    for i, pdb_path in enumerate(pdb_paths):
        if (i + 1) % 500 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(pdb_paths))

        stem = protein_stem(pdb_path.name)
        rec = {"pdb": pdb_path.name, "pdb_path": str(pdb_path),
               "protein": stem, "kept": False,
               "removal_reason": None, "rosetta_score": None, "n_residues": None,
               "max_gap": None, "max_repeat_run": None,
               "frac_ala": None, "frac_glu": None, "rmsd_vs_original": None}

        rec["n_residues"] = count_residues(pdb_path)
        if rec["n_residues"] == 0:
            rec["removal_reason"] = "parse_failed"
            records.append(rec)
            continue

        # Sequence checks
        seq = get_sequence(pdb_path)
        frac_a, frac_e = aa_fractions(seq)
        rec["frac_ala"] = frac_a
        rec["frac_glu"] = frac_e

        # Repeat residue check: flag proteins with >= 5 consecutive identical residues
        run_len = longest_repeat_run(seq)
        rec["max_repeat_run"] = run_len
        if run_len >= 5:
            score = minimize_and_score_pdb(pdb_path, scorefxn)
            rec["rosetta_score"] = score
            rec["removal_reason"] = "repeat_residues"
            records.append(rec)
            continue

        # Gap check: flag proteins with any gap > 1 in residue numbering
        gap = max_sequence_gap(pdb_path)
        rec["max_gap"] = gap
        if gap > 1:
            score = minimize_and_score_pdb(pdb_path, scorefxn)
            rec["rosetta_score"] = score
            rec["removal_reason"] = "sequence_gap"
            records.append(rec)
            continue

        # Rosetta score (minimize first, then filter on minimized energy)
        # Minimization follows run_backrub.py protocol: idealize, repack,
        # flip_HNQ, min chi, min chi+bb. See Smith & Kortemme (2008).
        score = minimize_and_score_pdb(pdb_path, scorefxn)
        rec["rosetta_score"] = score
        if score is None:
            rec["removal_reason"] = "score_failed"
            records.append(rec)
            continue
        # Only reject positive scores (unstable/unphysical). Lower scores are always better.
        if score > 0:
            rec["removal_reason"] = "score"
            records.append(rec)
            continue

        # Kabsch RMSD vs original structure
        if originals_dir is not None:
            if stem not in orig_coords_cache:
                orig_pdb = originals_dir / f"{stem}.pdb"
                if orig_pdb.exists():
                    _, coords = get_ca_atoms(orig_pdb)
                    orig_coords_cache[stem] = coords if len(coords) >= 3 else None
                else:
                    orig_coords_cache[stem] = None

            ref_coords = orig_coords_cache.get(stem)
            if ref_coords is not None:
                _, mob_coords = get_ca_atoms(pdb_path)
                rmsd = compute_rmsd_kabsch(ref_coords, mob_coords)
                rec["rmsd_vs_original"] = rmsd
                if not np.isnan(rmsd) and rmsd > rmsd_max:
                    rec["removal_reason"] = "rmsd"
                    records.append(rec)
                    continue
            # If original not found, skip RMSD check (don't penalize)

        rec["kept"] = True
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "filter_results.csv", index=False)

    # Write kept PDB list
    kept_df = df[df["kept"]]
    kept_list_path = output_dir / "kept_pdb_list.txt"
    kept_list_path.write_text("\n".join(kept_df["pdb_path"].tolist()) + "\n")

    # Summary
    total = len(df)
    print("\n" + "=" * 70)
    print("PDB DATASET QUALITY FILTER SUMMARY")
    print("=" * 70)
    print(f"  Total PDBs processed:    {total}")
    print(f"  Unique proteins:         {df['protein'].nunique()}")
    print(f"  Kept:                    {kept_df.shape[0]} ({kept_df.shape[0]/max(total,1)*100:.1f}%)")
    print(f"  Proteins with >= 1 kept: {kept_df['protein'].nunique()}")
    removed = df[~df["kept"]]
    if len(removed):
        print(f"  Removal breakdown:")
        for reason, n in removed["removal_reason"].value_counts().items():
            print(f"    {reason}: {n}")
    # Report fraction of ALL input PDBs with high Ala or Glu content
    all_ala = df["frac_ala"].dropna()
    all_glu = df["frac_glu"].dropna()
    if len(all_ala):
        n_ala_high = (all_ala > 0.20).sum()
        print(f"  Input Ala > 20%:         {n_ala_high}/{len(all_ala)} ({n_ala_high/len(all_ala)*100:.1f}%)")
    if len(all_glu):
        n_glu_high = (all_glu > 0.20).sum()
        print(f"  Input Glu > 20%:         {n_glu_high}/{len(all_glu)} ({n_glu_high/len(all_glu)*100:.1f}%)")

    if len(kept_df):
        scores = kept_df["rosetta_score"].dropna()
        print(f"  Kept score range:        [{scores.min():.1f}, {scores.max():.1f}]")
        print(f"  Kept score mean:         {scores.mean():.1f} +/- {scores.std():.1f}")
        res = kept_df["n_residues"].dropna()
        print(f"  Kept residue range:      [{res.min():.0f}, {res.max():.0f}]")
        ala = kept_df["frac_ala"].dropna()
        glu = kept_df["frac_glu"].dropna()
        print(f"  Kept Ala fraction:       {ala.mean():.4f} +/- {ala.std():.4f}")
        print(f"  Kept Glu fraction:       {glu.mean():.4f} +/- {glu.std():.4f}")
        rmsd_vals = kept_df["rmsd_vs_original"].dropna()
        if len(rmsd_vals):
            print(f"  Kept RMSD vs original:   {rmsd_vals.mean():.3f} +/- {rmsd_vals.std():.3f} A")
    print(f"  Kept list written to:    {kept_list_path}")
    print("=" * 70 + "\n")

    # Before/after plots
    plot_before_after(df, "rosetta_score", "Rosetta Score (REU)",
                      "#adb5bd", "#e76f51", figures_dir)
    plot_before_after(df, "n_residues", "Number of Residues",
                      "#adb5bd", "#457b9d", figures_dir)
    plot_before_after(df, "frac_ala", "Alanine Fraction",
                      "#adb5bd", "#2a9d8f", figures_dir)
    plot_before_after(df, "frac_glu", "Glutamate Fraction",
                      "#adb5bd", "#e9c46a", figures_dir)
    plot_before_after(df, "rmsd_vs_original", r"C$\alpha$ RMSD vs Original ($\AA$)",
                      "#adb5bd", "#264653", figures_dir)

    # Chi angle KDE: OG Ingraham originals vs augmented (before/after filter)
    if not skip_chi:
        logger.info("Generating chi angle KDE plots...")
        from collections import defaultdict
        from plot_chi_angle_kde import extract_chi_angles_from_pdb

        def extract_from_paths(paths, max_files=2000):
            all_angles = defaultdict(list)
            for p in paths[:max_files]:
                file_angles = extract_chi_angles_from_pdb(str(p))
                for key, vals in file_angles.items():
                    all_angles[key].extend(vals)
            return {k: np.array(v) for k, v in all_angles.items()}

        chi_data = {}

        # OG Ingraham CATH originals
        if originals_dir is not None:
            og_dir = Path(originals_dir)
            og_pdbs = sorted(og_dir.glob("*.pdb"))[:2000]
            if og_pdbs:
                logger.info("  Extracting chi angles from %d OG Ingraham originals...", len(og_pdbs))
                chi_data["OG Ingraham CATH"] = extract_from_paths(og_pdbs)

        # Augmented dataset (before filter)
        logger.info("  Extracting chi angles from augmented dataset (before filter)...")
        chi_data["Augmented (before)"] = extract_from_paths(pdb_paths)

        # After filter
        logger.info("  Extracting chi angles from augmented dataset (after filter)...")
        chi_data["Augmented (after)"] = extract_from_paths(
            [Path(p) for p in kept_df["pdb_path"].tolist()])

        for resname in AA_WITH_CHI:
            out_path = str(figures_dir / f"chi_angles_{resname}.png")
            plot_kde_for_amino_acid(resname, chi_data, out_path)

    logger.info("Done. Output: %s", output_dir)
    return df


# ── Ensemble mode (backrub conformers vs originals) ──────────────────────────

def run_ensemble_mode(data_dir, ensemble_dir, output_dir, figures_dir,
                      drmsd_max, skip_chi):
    """Filter backrub ensembles by minimized score (>0 removed), gaps, and DRMSD."""
    orig_dir = data_dir / "originals"

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
                   "max_gap": None,
                   "drmsd": None, "rmsd": None, "seq_identity": None, "n_residues": None}

            gap = max_sequence_gap(conf_pdb)
            rec["max_gap"] = gap
            if gap > 1:
                rec["removal_reason"] = "sequence_gap"
                records.append(rec)
                continue

            # Minimize before scoring (same protocol as run_backrub.py)
            score = minimize_and_score_pdb(conf_pdb, scorefxn)
            rec["rosetta_score"] = score
            # Only reject positive scores (unstable/unphysical). Lower is always better.
            if score is None or score > 0:
                rec["removal_reason"] = "score"
                records.append(rec)
                continue

            _, mob_coords = get_ca_atoms(conf_pdb)
            drmsd = compute_drmsd(ref_coords, mob_coords)
            rec["drmsd"] = drmsd
            rec["rmsd"] = compute_rmsd_kabsch(ref_coords, mob_coords)
            rec["n_residues"] = min(len(ref_coords), len(mob_coords))

            if np.isnan(drmsd) or drmsd > drmsd_max:
                rec["removal_reason"] = "drmsd"
                records.append(rec)
                continue

            rec["seq_identity"] = sequence_identity(orig_seq, get_sequence(conf_pdb))
            rec["kept"] = True
            records.append(rec)

            dst = output_dir / stem
            dst.mkdir(exist_ok=True)
            shutil.copy2(conf_pdb, dst / conf_pdb.name)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "filter_results.csv", index=False)

    # Summary
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

    # Before/after plots
    for col, label in [("rosetta_score", "Rosetta Score (REU)"),
                        ("drmsd", "Ca DRMSD (A)"),
                        ("rmsd", "Ca RMSD (A)"),
                        ("n_residues", "Number of Residues")]:
        plot_before_after(df, col, label, "#adb5bd", "#2a9d8f", figures_dir)

    # Chi angle KDE: before vs after
    if not skip_chi:
        logger.info("Generating chi angle KDE plots...")
        chi_data = {"Filtered": extract_chi_angles_from_dir(str(output_dir))}
        for resname in AA_WITH_CHI:
            out_path = str(figures_dir / f"chi_angles_{resname}.png")
            plot_kde_for_amino_acid(resname, chi_data, out_path)

    logger.info("Done. Output: %s", output_dir)
    return df


# Main

def main():
    parser = argparse.ArgumentParser(description="Filter PDB dataset by score, gaps, and DRMSD")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pdb-list", type=str,
                      help="Text file with one PDB path per line (flat list mode)")
    mode.add_argument("--data-dir", type=str,
                      help="Root dir with 'originals/' and ensemble subdirs (ensemble mode)")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                        help="Ensemble dir (default: <data-dir>/ai-cath_backrub_subset_ensembles)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dir for filtered results and plots")
    parser.add_argument("--originals-dir", type=str, default=None,
                        help="Dir with original PDBs (e.g. 1jvbA02.pdb) for RMSD comparison")
    parser.add_argument("--rmsd-max", type=float, default=3.0,
                        help="Max Kabsch RMSD vs original in Angstroms (default: 3.0)")
    parser.add_argument("--drmsd-max", type=float, default=10.0,
                        help="Max DRMSD for ensemble mode (default: 10.0)")
    parser.add_argument("--skip-chi", action="store_true", help="Skip chi angle KDE plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_list:
        run_pdb_list_mode(args.pdb_list, output_dir, figures_dir,
                          args.rmsd_max, args.originals_dir, args.skip_chi)
    else:
        data_dir = Path(args.data_dir)
        ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else data_dir / "ai-cath_backrub_subset_ensembles"
        run_ensemble_mode(data_dir, ensemble_dir, output_dir, figures_dir,
                          args.drmsd_max, args.skip_chi)


if __name__ == "__main__":
    main()
