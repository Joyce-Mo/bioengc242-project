#!/usr/bin/env python
"""Compute C alpha RMSD between original PDBs and their Backrub ensemble conformers.

For each protein in the originals directory, aligns each Backrub conformer
to the original using the Kabsch algorithm and reports the C alpha RMSD.

Outputs:
  - CSV with per-conformer RMSD values
  - Box plot of RMSD distributions per protein
  - Summary statistics printed to stdout

Usage:
    python scripts/rmsd_backrub_vs_original.py \
        --data-dir /path/to/ai-cath_subset \
        --output-dir output/backrub_rmsd
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_ca_atoms(pdb_path):
    """Extract Cα atoms from the first model/chain of a PDB file.

    Returns a list of Bio.PDB Atom objects (needed for Superimposer)
    and the corresponding coordinate array.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    atoms.append(residue["CA"])
        break  # first model only
    coords = np.array([a.get_vector().get_array() for a in atoms])
    return atoms, coords


def compute_rmsd_kabsch(coords_ref, coords_mob):
    """Compute RMSD after optimal superposition (Kabsch algorithm).

    Uses the minimum of len(coords_ref) and len(coords_mob) residues.
    """
    min_len = min(len(coords_ref), len(coords_mob))
    if min_len < 3:
        return np.nan

    ref = coords_ref[:min_len]
    mob = coords_mob[:min_len]

    # Center both
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref_c = ref - ref_center
    mob_c = mob - mob_center

    # Kabsch: find optimal rotation via SVD
    H = mob_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    mob_aligned = (mob_c @ R.T)
    rmsd = np.sqrt(np.mean(np.sum((ref_c - mob_aligned) ** 2, axis=1)))
    return rmsd


def compute_drmsd(coords_ref, coords_mob):
    """Compute distance RMSD (DRMSD) — alignment-free.

    DRMSD = sqrt(mean((d_ref(i,j) - d_mob(i,j))^2)) over all Cα pairs.
    Captures internal geometry differences without superposition.
    """
    min_len = min(len(coords_ref), len(coords_mob))
    if min_len < 3:
        return np.nan

    ref = coords_ref[:min_len]
    mob = coords_mob[:min_len]

    # Intra-protein pairwise Cα distances
    diff_ref = ref[:, np.newaxis, :] - ref[np.newaxis, :, :]
    dist_ref = np.sqrt(np.sum(diff_ref ** 2, axis=-1))

    diff_mob = mob[:, np.newaxis, :] - mob[np.newaxis, :, :]
    dist_mob = np.sqrt(np.sum(diff_mob ** 2, axis=-1))

    # Upper triangle only (avoid double-counting and diagonal zeros)
    triu_idx = np.triu_indices(min_len, k=1)
    diff_sq = (dist_ref[triu_idx] - dist_mob[triu_idx]) ** 2
    return np.sqrt(np.mean(diff_sq))


def main():
    parser = argparse.ArgumentParser(
        description="Cα RMSD: Backrub conformers vs. original structures"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory containing 'originals/' and 'ai-cath_backrub_subset_ensembles/' subdirs",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/backrub_rmsd",
        help="Directory for output CSV and figures (default: output/backrub_rmsd)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    orig_dir = data_dir / "originals"
    ensemble_dir = data_dir / "ai-cath_backrub_subset_ensembles"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not orig_dir.is_dir():
        logger.error("Originals directory not found: %s", orig_dir)
        sys.exit(1)
    if not ensemble_dir.is_dir():
        logger.error("Backrub ensembles directory not found: %s", ensemble_dir)
        sys.exit(1)

    # Collect all original PDBs
    orig_pdbs = sorted(orig_dir.glob("*.pdb"))
    logger.info("Found %d original PDB files", len(orig_pdbs))

    records = []
    for orig_pdb in orig_pdbs:
        stem = orig_pdb.stem
        conformer_dir = ensemble_dir / stem

        if not conformer_dir.is_dir():
            logger.warning("No backrub ensemble directory for %s, skipping", stem)
            continue

        _, ref_coords = get_ca_atoms(orig_pdb)
        conformers = sorted(conformer_dir.glob("*.pdb"))

        for conf_pdb in conformers:
            _, mob_coords = get_ca_atoms(conf_pdb)
            rmsd = compute_rmsd_kabsch(ref_coords, mob_coords)
            drmsd = compute_drmsd(ref_coords, mob_coords)
            records.append({
                "protein": stem,
                "conformer": conf_pdb.stem,
                "rmsd": rmsd,
                "drmsd": drmsd,
                "n_residues": min(len(ref_coords), len(mob_coords)),
            })
            logger.info("  %s vs %s: RMSD = %.3f Å, DRMSD = %.3f Å (%d residues)",
                        stem, conf_pdb.stem, rmsd, drmsd,
                        min(len(ref_coords), len(mob_coords)))

    if not records:
        logger.error("No conformer comparisons found.")
        sys.exit(1)

    df = pd.DataFrame(records)
    csv_path = output_dir / "backrub_rmsd.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved RMSD table: %s", csv_path)

    #  Summary statistics   
    summary = df.groupby("protein")["rmsd"].agg(["mean", "std", "min", "max", "count"])
    summary_path = output_dir / "backrub_rmsd_summary.csv"
    summary.to_csv(summary_path)
    print("\n" + "=" * 70)
    print("BACKRUB vs ORIGINAL — RMSD SUMMARY (Å)")
    print("=" * 70)
    print(summary.round(3).to_string())
    print(f"\nOverall mean RMSD: {df['rmsd'].mean():.3f} Å")
    print(f"Overall std RMSD:  {df['rmsd'].std():.3f} Å")
    print("=" * 70 + "\n")

    #  DRMSD summary   
    drmsd_summary = df.groupby("protein")["drmsd"].agg(["mean", "std", "min", "max", "count"])
    drmsd_summary_path = output_dir / "backrub_drmsd_summary.csv"
    drmsd_summary.to_csv(drmsd_summary_path)
    print("=" * 70)
    print("BACKRUB vs ORIGINAL — Cα DRMSD SUMMARY (Å)")
    print("=" * 70)
    print(drmsd_summary.round(3).to_string())
    print(f"\nOverall mean DRMSD: {df['drmsd'].mean():.3f} Å")
    print(f"Overall std DRMSD:  {df['drmsd'].std():.3f} Å")
    print("=" * 70 + "\n")

    #  Box plot per protein   
    proteins_sorted = df.groupby("protein")["rmsd"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(proteins_sorted) * 0.4), 6))
    df_plot = df.set_index("protein").loc[proteins_sorted].reset_index()
    bp = ax.boxplot(
        [df_plot[df_plot["protein"] == p]["rmsd"].values for p in proteins_sorted],
        labels=proteins_sorted,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xticklabels(proteins_sorted, rotation=90, fontsize=7)
    ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("Backrub Conformer RMSD vs. Original Structure")
    fig.tight_layout()
    fig.savefig(figures_dir / "backrub_rmsd_boxplot.png", dpi=150)
    plt.close(fig)

    #  DRMSD box plot per protein   
    proteins_sorted_d = df.groupby("protein")["drmsd"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(proteins_sorted_d) * 0.4), 6))
    df_plot_d = df.set_index("protein").loc[proteins_sorted_d].reset_index()
    bp = ax.boxplot(
        [df_plot_d[df_plot_d["protein"] == p]["drmsd"].values for p in proteins_sorted_d],
        labels=proteins_sorted_d,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("coral")
        patch.set_alpha(0.7)

    ax.set_xticklabels(proteins_sorted_d, rotation=90, fontsize=7)
    ax.set_ylabel("Cα DRMSD (Å)")
    ax.set_title("Backrub Conformer DRMSD vs. Original Structure")
    fig.tight_layout()
    fig.savefig(figures_dir / "backrub_drmsd_boxplot.png", dpi=150)
    plt.close(fig)

    #  Histogram of all RMSDs   
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["rmsd"].dropna(), bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(df["rmsd"].mean(), color="red", linestyle="--", label=f"Mean = {df['rmsd'].mean():.2f} Å")
    ax.set_xlabel("Cα RMSD (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Backrub Conformer RMSD (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "backrub_rmsd_histogram.png", dpi=150)
    plt.close(fig)

    #  Scatter: RMSD vs protein length   
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["n_residues"], df["rmsd"], s=20, alpha=0.6, c="steelblue")
    ax.set_xlabel("Number of residues")
    ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("Backrub RMSD vs. Protein Length")
    fig.tight_layout()
    fig.savefig(figures_dir / "backrub_rmsd_vs_length.png", dpi=150)
    plt.close(fig)

    logger.info("All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
