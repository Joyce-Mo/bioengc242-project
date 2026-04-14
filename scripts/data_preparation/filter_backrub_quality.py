#!/usr/bin/env python
"""Filter PDB datasets by Rosetta score, sequence gaps, and (optionally) DRMSD.

Two modes:
  --pdb-list FILE   Filter a flat list of PDBs (score + gap checks only).
                    Writes kept paths to <output-dir>/kept_pdb_list.txt.
  --data-dir DIR    Filter backrub ensembles vs originals (score + gap + DRMSD).

Criteria:
  1. Rosetta score > 0 or outside [score-min, score-max] -> remove
  2. Sequence gaps in residue numbering -> remove
  3. DRMSD > drmsd-max vs original -> remove (ensemble mode only)

Outputs filter_results.csv and before/after comparison plots.

Usage:
    # Filter a flat PDB list (pre-backrub dataset validation)
    python scripts/data_preparation/filter_backrub_quality.py \
        --pdb-list ai-cath_training_pdb.txt --output-dir /path/to/filtered

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


# Sequence gap detection 

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


def has_repeat_residues(seq, min_run=3):
    """Check if sequence has any run of >= min_run identical residues."""
    if len(seq) < min_run:
        return False
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
            if count >= min_run:
                return True
        else:
            count = 1
    return False


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
                      score_min, score_max, drmsd_max, skip_chi):
    """Filter a flat PDB list by Rosetta score, gaps, repeats, and intra-protein DRMSD."""
    from collections import defaultdict as _defaultdict

    with open(pdb_list_path) as f:
        pdb_paths = [Path(line.strip()) for line in f if line.strip()]
    logger.info("Loaded %d PDB paths from %s", len(pdb_paths), pdb_list_path)

    # Group conformers by protein stem (e.g. 1jvbA02_0.pdb -> 1jvbA02)
    protein_groups = _defaultdict(list)
    for p in pdb_paths:
        protein_groups[protein_stem(p.name)].append(p)
    logger.info("Found %d unique proteins", len(protein_groups))

    # Compute intra-protein DRMSD (max pairwise DRMSD within each protein's conformers)
    logger.info("Computing intra-protein pairwise DRMSD...")
    all_max_drmsd = {}  # pdb_path_str -> max DRMSD to sibling
    for prot_id, paths in protein_groups.items():
        if len(paths) < 2:
            for p in paths:
                all_max_drmsd[str(p)] = 0.0
            continue
        drmsd_map = compute_intra_protein_drmsd(paths)
        all_max_drmsd.update(drmsd_map)

    scorefxn = init_rosetta()
    records = []

    for i, pdb_path in enumerate(pdb_paths):
        if (i + 1) % 500 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(pdb_paths))

        rec = {"pdb": pdb_path.name, "pdb_path": str(pdb_path),
               "protein": protein_stem(pdb_path.name), "kept": False,
               "removal_reason": None, "rosetta_score": None, "n_residues": None,
               "has_gap": None, "has_repeats": None,
               "frac_ala": None, "frac_glu": None, "max_drmsd": None}

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

        # Repeat residue check
        repeats = has_repeat_residues(seq, min_run=3)
        rec["has_repeats"] = repeats
        if repeats:
            score = score_pdb(pdb_path, scorefxn)
            rec["rosetta_score"] = score
            rec["removal_reason"] = "repeat_residues"
            records.append(rec)
            continue

        # Gap check
        gap = has_sequence_gaps(pdb_path)
        rec["has_gap"] = gap
        if gap:
            score = score_pdb(pdb_path, scorefxn)
            rec["rosetta_score"] = score
            rec["removal_reason"] = "sequence_gap"
            records.append(rec)
            continue

        # Rosetta score
        score = score_pdb(pdb_path, scorefxn)
        rec["rosetta_score"] = score
        if score is None:
            rec["removal_reason"] = "score_failed"
            records.append(rec)
            continue
        if score > 0 or score < score_min or score > score_max:
            rec["removal_reason"] = "score"
            records.append(rec)
            continue

        # Intra-protein DRMSD check
        max_d = all_max_drmsd.get(str(pdb_path), 0.0)
        rec["max_drmsd"] = max_d
        if max_d > drmsd_max:
            rec["removal_reason"] = "drmsd"
            records.append(rec)
            continue

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
        drmsd_vals = kept_df["max_drmsd"].dropna()
        if len(drmsd_vals):
            print(f"  Kept max DRMSD:          {drmsd_vals.mean():.3f} +/- {drmsd_vals.std():.3f} A")
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
    plot_before_after(df, "max_drmsd", "Max Intra-Protein DRMSD (A)",
                      "#adb5bd", "#264653", figures_dir)

    # Chi angle KDE: before vs after
    if not skip_chi:
        logger.info("Generating chi angle KDE plots (before vs after)...")
        from collections import defaultdict
        from plot_chi_angle_kde import extract_chi_angles_from_pdb

        def extract_from_paths(paths, max_files=2000):
            all_angles = defaultdict(list)
            for p in paths[:max_files]:
                file_angles = extract_chi_angles_from_pdb(str(p))
                for key, vals in file_angles.items():
                    all_angles[key].extend(vals)
            return {k: np.array(v) for k, v in all_angles.items()}

        before_chi = extract_from_paths(pdb_paths)
        after_chi = extract_from_paths([Path(p) for p in kept_df["pdb_path"].tolist()])
        chi_data = {"Before filter": before_chi, "After filter": after_chi}

        for resname in AA_WITH_CHI:
            out_path = str(figures_dir / f"chi_angles_{resname}.png")
            plot_kde_for_amino_acid(resname, chi_data, out_path)

    logger.info("Done. Output: %s", output_dir)
    return df


# ── Ensemble mode (backrub conformers vs originals) ──────────────────────────

def run_ensemble_mode(data_dir, ensemble_dir, output_dir, figures_dir,
                      score_min, score_max, drmsd_max, skip_chi):
    """Filter backrub ensembles by score, gaps, and DRMSD."""
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
                   "drmsd": None, "rmsd": None, "seq_identity": None, "n_residues": None}

            if has_sequence_gaps(conf_pdb):
                rec["removal_reason"] = "sequence_gap"
                records.append(rec)
                continue

            score = score_pdb(conf_pdb, scorefxn)
            rec["rosetta_score"] = score
            if score is None or score > 0 or score < score_min or score > score_max:
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
    parser.add_argument("--score-min", type=float, default=-1200)
    parser.add_argument("--score-max", type=float, default=-0)
    parser.add_argument("--drmsd-max", type=float, default=10.0)
    parser.add_argument("--skip-chi", action="store_true", help="Skip chi angle KDE plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_list:
        run_pdb_list_mode(args.pdb_list, output_dir, figures_dir,
                          args.score_min, args.score_max, args.drmsd_max, args.skip_chi)
    else:
        data_dir = Path(args.data_dir)
        ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else data_dir / "ai-cath_backrub_subset_ensembles"
        run_ensemble_mode(data_dir, ensemble_dir, output_dir, figures_dir,
                          args.score_min, args.score_max, args.drmsd_max, args.skip_chi)


if __name__ == "__main__":
    main()
