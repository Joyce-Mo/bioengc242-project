"""Foldseek structural analysis of a protein PDB dataset.

Runs foldseek all-vs-all structural alignment on a directory of PDB files
and generates the following outputs:

Figures:
  - Position-by-position alignment quality heatmap
  - TM-score distribution and pairwise heatmap
  - RMSD distribution and pairwise heatmap
  - Per-protein residue alignment coverage bar chart
  - Pairwise Cα DRMSD (distance-RMSD) matrix heatmap
  - UMAP embedding of structural similarity
  - Multidimensional Scaling (MDS) embedding of structural similarity

Tables:
  - Correlation table between structural metrics (TM-score, RMSD, seq identity, etc.)

Usage:
    python scripts/foldseek_analysis.py --pdb-dir /path/to/pdbs --output-dir output/foldseek
    python scripts/foldseek_analysis.py --pdb-dir /path/to/pdbs --output-dir output/foldseek --threads 8
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser
from sklearn.manifold import MDS
from umap import UMAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Set paths — defaults point to the foldseek-filtered dataset produced by
# foldseek_tm_filter.py, so this script focuses on visualization/analysis.
# ---------------------------------------------------------------------------
DEFAULT_PDB_DIR = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered_foldseek_filtered"
DEFAULT_OUTPUT_DIR = "output/Karson_data_foldseek"
DEFAULT_RESULTS_TSV = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered_foldseek_filtered/foldseek_results.tsv"

# ---------------------------------------------------------------------------
# Foldseek output columns for --format-output
# ---------------------------------------------------------------------------

# Fields we request from foldseek easy-search
FOLDSEEK_COLUMNS = [
    "query",       # query protein name
    "target",      # target protein name
    "fident",      # fractional sequence identity
    "alnlen",      # alignment length
    "mismatch",    # number of mismatches
    "gapopen",     # number of gap openings
    "qstart",      # query alignment start position
    "qend",        # query alignment end position
    "tstart",      # target alignment start position
    "tend",        # target alignment end position
    "evalue",      # E-value
    "bits",        # bit score
    "alntmscore",  # TM-score of the alignment
    "rmsd",        # RMSD of the alignment (Angstroms)
    "qaln",        # query alignment string
    "taln",        # target alignment string
    "qlen",        # query sequence length
    "tlen",        # target sequence length
]

FOLDSEEK_FORMAT_STR = ",".join(FOLDSEEK_COLUMNS)


# ---------------------------------------------------------------------------
# Step 1: Run foldseek all-vs-all
# ---------------------------------------------------------------------------

def run_foldseek(pdb_dir, output_dir, threads=4):
    """Run foldseek easy-search in all-vs-all mode on the PDB directory.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB files.
    output_dir : Path
        Directory for foldseek output files.
    threads : int
        Number of threads for foldseek.

    Returns
    -------
    Path
        Path to the foldseek results TSV file.
    """
    results_path = output_dir / "foldseek_results.tsv"

    # Use a temp directory for foldseek's internal databases
    tmp_dir = output_dir / "foldseek_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        str(pdb_dir),           # query: entire directory
        str(pdb_dir),           # target: same directory (all-vs-all)
        str(results_path),      # output alignments
        str(tmp_dir),           # temp directory for DBs
        "--format-output", FOLDSEEK_FORMAT_STR,
        "--threads", str(threads),
        "--exhaustive-search",  # ensures all-vs-all, no prefilter skipping
        "--tmscore-threshold", "0.5",  # only report alignments with TM-score > 0.5
    ]

    logger.info("Running foldseek: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Foldseek stderr:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Foldseek finished. Results at %s", results_path)
    return results_path


# ---------------------------------------------------------------------------
# Step 2: Parse results
# ---------------------------------------------------------------------------

def load_results(results_path):
    """Load foldseek results TSV into a DataFrame.

    Parameters
    ----------
    results_path : Path
        Path to the foldseek output TSV.

    Returns
    -------
    pd.DataFrame
        Parsed alignment results with typed columns.
    """
    df = pd.read_csv(results_path, sep="\t", header=None, names=FOLDSEEK_COLUMNS)

    # Strip any file extensions from protein names for cleaner labels
    df["query"] = df["query"].astype(str).str.replace(r"\.pdb$", "", regex=True)
    df["target"] = df["target"].astype(str).str.replace(r"\.pdb$", "", regex=True)

    logger.info("Loaded %d alignments between %d unique proteins",
                len(df), df["query"].nunique())
    return df


# ---------------------------------------------------------------------------
# Step 3: Build pairwise matrices
# ---------------------------------------------------------------------------

def build_pairwise_matrix(df, metric, proteins):
    """Build a symmetric pairwise matrix from alignment results.

    For each (query, target) pair, takes the best (max TM-score / min RMSD)
    alignment if multiple exist. Self-comparisons are set to the identity
    value (1.0 for TM-score/fident, 0.0 for RMSD).

    Parameters
    ----------
    df : pd.DataFrame
        Foldseek alignment results.
    metric : str
        Column name to use as the pairwise value (e.g. 'alntmscore', 'rmsd').
    proteins : list[str]
        Ordered list of protein names for matrix rows/columns.

    Returns
    -------
    np.ndarray
        Square matrix of shape (n_proteins, n_proteins).
    """
    n = len(proteins)
    prot_idx = {p: i for i, p in enumerate(proteins)}

    # Set default fill: 0 for similarity metrics, NaN for distance metrics
    if metric == "rmsd":
        mat = np.full((n, n), np.nan)
        np.fill_diagonal(mat, 0.0)
    else:
        mat = np.zeros((n, n))
        np.fill_diagonal(mat, 1.0)

    # Group by pair and take best alignment
    if metric == "rmsd":
        # For RMSD, best = minimum
        best = df.groupby(["query", "target"])[metric].min()
    else:
        # For similarity metrics, best = maximum
        best = df.groupby(["query", "target"])[metric].max()

    for (q, t), val in best.items():
        if q in prot_idx and t in prot_idx:
            i, j = prot_idx[q], prot_idx[t]
            mat[i, j] = val
            mat[j, i] = val  # symmetrize

    return mat


# ---------------------------------------------------------------------------
# Step 4: Compute pairwise Cα DRMSD
# ---------------------------------------------------------------------------

def _get_ca_coords(pdb_path):
    """Extract Cα coordinates from the first model/chain of a PDB file.

    Parameters
    ----------
    pdb_path : Path
        Path to a PDB file.

    Returns
    -------
    np.ndarray
        Array of shape (n_residues, 3) with Cα xyz coordinates.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only standard residues
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
        break  # first model only
    return np.array(coords)


def _intra_ca_distance_matrix(coords):
    """Compute the intra-protein pairwise Cα distance matrix.

    Parameters
    ----------
    coords : np.ndarray
        Shape (n_residues, 3) array of Cα coordinates.

    Returns
    -------
    np.ndarray
        Shape (n_residues, n_residues) symmetric distance matrix.
    """
    # diff[i, j] = coords[i] - coords[j], shape (n, n, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def compute_drmsd(coords_a, coords_b):
    """Compute DRMSD between two structures.

    DRMSD = sqrt( mean( (d_A(i,j) - d_B(i,j))^2 ) ) over all residue pairs,
    where d_X(i,j) is the Cα-Cα distance between residues i and j in
    structure X. Only uses the first min(len_a, len_b) residues.

    Parameters
    ----------
    coords_a : np.ndarray
        Shape (n_a, 3) Cα coordinates for protein A.
    coords_b : np.ndarray
        Shape (n_b, 3) Cα coordinates for protein B.

    Returns
    -------
    float
        DRMSD value, or np.nan if alignment is too short (< 3 residues).
    """
    # Truncate to the shorter length (simple positional alignment)
    min_len = min(len(coords_a), len(coords_b))
    if min_len < 3:
        return np.nan

    dist_a = _intra_ca_distance_matrix(coords_a[:min_len])
    dist_b = _intra_ca_distance_matrix(coords_b[:min_len])

    # Upper triangle only (avoid double-counting and diagonal zeros)
    triu_idx = np.triu_indices(min_len, k=1)
    diff_sq = (dist_a[triu_idx] - dist_b[triu_idx]) ** 2
    return np.sqrt(np.mean(diff_sq))


def compute_pairwise_drmsd(pdb_dir, proteins):
    """Compute pairwise DRMSD matrix across all proteins.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB files.
    proteins : list[str]
        Ordered list of protein names (without .pdb extension).

    Returns
    -------
    np.ndarray
        Shape (n, n) symmetric DRMSD matrix.
    """
    n = len(proteins)
    logger.info("Computing pairwise Cα DRMSD for %d proteins...", n)

    # Pre-load all Cα coordinates
    all_coords = {}
    for prot in proteins:
        pdb_path = pdb_dir / f"{prot}.pdb"
        if pdb_path.exists():
            try:
                all_coords[prot] = _get_ca_coords(pdb_path)
            except Exception as e:
                logger.warning("Could not extract Cα coords from %s: %s", prot, e)

    drmsd_mat = np.full((n, n), np.nan)
    np.fill_diagonal(drmsd_mat, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            if proteins[i] in all_coords and proteins[j] in all_coords:
                val = compute_drmsd(all_coords[proteins[i]], all_coords[proteins[j]])
                drmsd_mat[i, j] = val
                drmsd_mat[j, i] = val

    return drmsd_mat


# ---------------------------------------------------------------------------
# Step 5: Figures
# ---------------------------------------------------------------------------

def plot_metric_distribution(df, metric, label, units, output_path):
    """Plot violin + histogram of a structural metric (excluding self-hits).

    Produces a two-panel figure:
      - Top: seaborn histogram (with KDE) for overall distribution
      - Bottom: violin plots per protein (if <=20 proteins) or a single
        violin with strip overlay

    Parameters
    ----------
    df : pd.DataFrame
        Foldseek alignment results.
    metric : str
        Column name to plot.
    label : str
        Human-readable label for the metric.
    units : str
        Units string for axis label.
    output_path : Path
        File path to save the figure.
    """
    TEAL = "#2a9d8f"

    # Exclude self-comparisons
    non_self = df.loc[df["query"] != df["target"]].dropna(subset=[metric])
    vals = non_self[metric]

    sns.set_theme(style="whitegrid", palette="muted")

    n_proteins = non_self["query"].nunique()

    if n_proteins <= 20:
        # Two-panel: histogram on top, per-protein violins on bottom
        fig, (ax_hist, ax_violin) = plt.subplots(
            2, 1, figsize=(8, max(6, 2.5 + n_proteins * 0.4)),
            gridspec_kw={"height_ratios": [1, max(1, n_proteins * 0.15)]},
        )

        # Histogram panel
        sns.histplot(vals, bins=50, color=TEAL, edgecolor="white",
                     linewidth=0.5, kde=True, ax=ax_hist)
        ax_hist.set_xlabel(f"{label} ({units})")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Distribution of {label} (n={len(vals)} alignments)")

        # Per-protein violins (horizontal), sorted by median
        order = (non_self.groupby("query")[metric]
                 .median().sort_values(ascending=False).index.tolist())
        sns.violinplot(
            data=non_self, y="query", x=metric, order=order,
            orient="h", inner=None, linewidth=0.8, saturation=0.8,
            color=TEAL, cut=0, ax=ax_violin,
        )
        sns.stripplot(
            data=non_self, y="query", x=metric, order=order,
            orient="h", size=2, alpha=0.4, color="0.2", jitter=True,
            ax=ax_violin,
        )
        ax_violin.set_xlabel(f"{label} ({units})")
        ax_violin.set_ylabel("")
    else:
        # Two-panel: histogram on top, single violin on bottom
        fig, (ax_hist, ax_violin) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]},
        )

        # Histogram panel
        sns.histplot(vals, bins=50, color=TEAL, edgecolor="white",
                     linewidth=0.5, kde=True, ax=ax_hist)
        ax_hist.set_xlabel(f"{label} ({units})")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Distribution of {label} (n={len(vals)} alignments)")

        # Single violin with strip overlay
        sns.violinplot(
            data=non_self, x=metric, inner="quartile",
            linewidth=0.8, color=TEAL, cut=0, ax=ax_violin,
        )
        sns.stripplot(
            data=non_self, x=metric, size=1.5, alpha=0.15,
            color="0.2", jitter=True, ax=ax_violin,
        )
        ax_violin.set_xlabel(f"{label} ({units})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    # Reset seaborn to avoid leaking style into other plots
    sns.reset_defaults()
    logger.info("Saved %s distribution: %s", label, output_path)


def plot_pairwise_heatmap(matrix, proteins, metric_label, output_path,
                          cmap="viridis", vmin=None, vmax=None):
    """Plot a pairwise heatmap from a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square (n, n) matrix of pairwise values.
    proteins : list[str]
        Protein names for axis labels.
    metric_label : str
        Label for the colorbar.
    output_path : Path
        File path to save the figure.
    cmap : str
        Matplotlib colormap name.
    vmin : float or None
        Colorbar minimum.
    vmax : float or None
        Colorbar maximum.
    """
    n = len(proteins)
    # For large datasets, suppress tick labels to keep the figure readable
    show_labels = n <= 50

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=metric_label)

    if show_labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(proteins, rotation=90, fontsize=6)
        ax.set_yticklabels(proteins, fontsize=6)
    else:
        ax.set_xlabel("Protein index")
        ax.set_ylabel("Protein index")

    ax.set_title(f"Pairwise {metric_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved pairwise heatmap: %s", output_path)


def plot_position_alignment_heatmap(df, output_path, max_proteins=100):
    """Plot a position-by-position alignment quality heatmap.

    For each protein (as query), computes the fraction of targets that
    align at each residue position. Shows which positions are structurally
    conserved across the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Foldseek results with 'qaln' and 'qlen' columns.
    output_path : Path
        File path to save the figure.
    max_proteins : int
        Maximum number of proteins to display (for readability).
    """
    proteins = sorted(df["query"].unique())[:max_proteins]
    subset = df[df["query"].isin(proteins) & (df["query"] != df["target"])].dropna(subset=["qlen", "qaln"])

    if subset.empty:
        logger.warning("No non-self alignments found for position heatmap.")
        return

    # For each query protein, build a position-wise alignment coverage vector
    max_len = int(subset["qlen"].max())
    coverage = np.zeros((len(proteins), max_len))

    for idx, prot in enumerate(proteins):
        prot_df = subset[subset["query"] == prot]
        if prot_df.empty:
            continue
        n_targets = len(prot_df)

        # For each alignment, mark aligned positions
        for _, row in prot_df.iterrows():
            qstart = int(row["qstart"]) - 1  # 0-indexed
            qaln = str(row["qaln"])
            pos = qstart
            for char in qaln:
                if char != "-" and pos < max_len:
                    coverage[idx, pos] += 1
                    pos += 1
                elif char == "-":
                    continue  # gap in query

        # Normalize by number of target alignments
        coverage[idx, :] /= max(n_targets, 1)

    fig, ax = plt.subplots(figsize=(12, max(6, len(proteins) * 0.15)))
    im = ax.imshow(coverage, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Alignment coverage (fraction of targets)")
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Protein")

    if len(proteins) <= 50:
        ax.set_yticks(range(len(proteins)))
        ax.set_yticklabels(proteins, fontsize=6)

    ax.set_title("Position-by-Position Alignment Coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved position alignment heatmap: %s", output_path)


def plot_residue_alignment_per_protein(df, output_path, max_proteins=50):
    """Plot per-protein residue alignment coverage as a bar chart.

    Shows the mean fraction of residues aligned per protein (as query)
    across all target comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Foldseek alignment results.
    output_path : Path
        File path to save the figure.
    max_proteins : int
        Maximum number of proteins to show.
    """
    # Exclude self-comparisons
    non_self = df[df["query"] != df["target"]].copy()
    # Fraction of query residues aligned = alnlen / qlen
    non_self["aln_frac"] = non_self["alnlen"] / non_self["qlen"]

    mean_cov = non_self.groupby("query")["aln_frac"].mean().sort_values(ascending=False)
    mean_cov = mean_cov.head(max_proteins)

    fig, ax = plt.subplots(figsize=(12, max(5, len(mean_cov) * 0.25)))
    ax.barh(range(len(mean_cov)), mean_cov.values, color="steelblue")
    ax.set_yticks(range(len(mean_cov)))
    ax.set_yticklabels(mean_cov.index, fontsize=7)
    ax.set_xlabel("Mean fraction of residues aligned")
    ax.set_title("Per-Protein Residue Alignment Coverage")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residue alignment bar chart: %s", output_path)


def plot_embedding(matrix, proteins, method_name, output_path):
    """Plot a 2D embedding (UMAP or MDS) colored by cluster.

    Parameters
    ----------
    matrix : np.ndarray
        Square distance/dissimilarity matrix of shape (n, n).
    proteins : list[str]
        Protein names for hover labels.
    method_name : str
        Name of the method ('UMAP' or 'MDS') for the title.
    output_path : Path
        File path to save the figure.
    """
    # Replace NaNs with the maximum observed value (treat missing as max distance)
    mat_clean = matrix.copy()
    max_val = np.nanmax(mat_clean)
    mat_clean[np.isnan(mat_clean)] = max_val

    n = len(proteins)

    if method_name == "UMAP":
        # UMAP on precomputed distance matrix
        n_neighbors = min(15, n - 1)
        reducer = UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=n_neighbors,
            random_state=42,
        )
        coords = reducer.fit_transform(mat_clean)
    else:
        # MDS on precomputed dissimilarity matrix
        reducer = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
        )
        coords = reducer.fit_transform(mat_clean)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7, c="steelblue")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"{method_name} Embedding of Structural Similarity (n={n})")

    # Label points if few enough
    if n <= 30:
        for i, prot in enumerate(proteins):
            ax.annotate(prot, (coords[i, 0], coords[i, 1]),
                        fontsize=5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s embedding: %s", method_name, output_path)


# ---------------------------------------------------------------------------
# Step 6: Correlation table
# ---------------------------------------------------------------------------

def compute_correlation_table(df, output_path):
    """Compute and save a correlation table between structural metrics.

    Correlates: TM-score, RMSD, sequence identity, alignment length,
    E-value, and bit score across all non-self alignments.

    Parameters
    ----------
    df : pd.DataFrame
        Foldseek alignment results.
    output_path : Path
        File path to save the correlation table CSV.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    non_self = df[df["query"] != df["target"]]

    metrics = {
        "TM-score": "alntmscore",
        "RMSD": "rmsd",
        "Seq Identity": "fident",
        "Alignment Length": "alnlen",
        "E-value": "evalue",
        "Bit Score": "bits",
    }

    metric_df = non_self[list(metrics.values())].rename(
        columns={v: k for k, v in metrics.items()}
    )

    corr = metric_df.corr(method="pearson")

    # Save as CSV
    corr.to_csv(output_path)
    logger.info("Saved correlation table: %s", output_path)

    # Also print it
    print("\n" + "=" * 60)
    print("METRIC CORRELATION TABLE (Pearson)")
    print("=" * 60)
    print(corr.round(3).to_string())
    print("=" * 60 + "\n")

    return corr


def plot_correlation_heatmap(corr, output_path):
    """Plot the correlation matrix as a heatmap.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix from compute_correlation_table().
    output_path : Path
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                linewidths=0.5)
    ax.set_title("Structural Metric Correlations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved correlation heatmap: %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Entry point for foldseek structural analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Foldseek structural analysis: alignments, DRMSD, UMAP, MDS, correlations.",
    )
    parser.add_argument(
        "--pdb-dir", type=str, default=DEFAULT_PDB_DIR,
        help=f"Directory containing PDB files to analyze (default: {DEFAULT_PDB_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output figures and tables (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--results-tsv", type=str, default=DEFAULT_RESULTS_TSV,
        help=f"Path to pre-computed foldseek results TSV (default: {DEFAULT_RESULTS_TSV}). "
             "If provided and exists, skips running foldseek.",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of threads for foldseek (default: 4).",
    )
    parser.add_argument(
        "--run-foldseek", action="store_true",
        help="Force running foldseek even if a results TSV already exists.",
    )
    parser.add_argument(
        "--skip-drmsd", action="store_true",
        help="Skip pairwise DRMSD computation (slow for large datasets).",
    )
    parser.add_argument(
        "--max-proteins-heatmap", type=int, default=200,
        help="Max proteins to show in heatmaps (default: 200).",
    )
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not pdb_dir.is_dir():
        logger.error("PDB directory does not exist: %s", pdb_dir)
        sys.exit(1)

    # ---- Step 1: Load or run foldseek ----
    # By default, use the pre-computed results TSV from foldseek_tm_filter.py.
    # Only run foldseek if --run-foldseek is explicitly passed.
    results_path = Path(args.results_tsv)
    if not args.run_foldseek and results_path.exists():
        logger.info("Using pre-computed foldseek results: %s", results_path)
    elif args.run_foldseek:
        results_path = run_foldseek(pdb_dir, output_dir, args.threads)
    else:
        logger.error("Results TSV not found: %s. Run foldseek_tm_filter.py first, "
                      "or pass --run-foldseek to run it here.", results_path)
        sys.exit(1)

    # ---- Step 2: Load results ----
    df = load_results(results_path)

    if df.empty:
        logger.error("No alignments found in foldseek results. Exiting.")
        sys.exit(1)

    # Get sorted list of unique proteins (cap for heatmaps)
    all_proteins = sorted(df["query"].unique())
    heatmap_proteins = all_proteins[:args.max_proteins_heatmap]
    logger.info("Total unique proteins: %d (showing %d in heatmaps)",
                len(all_proteins), len(heatmap_proteins))

    # ---- Step 3: Distributions ----
    plot_metric_distribution(
        df, "alntmscore", "TM-score", "0-1",
        figures_dir / "tmscore_distribution.png",
    )
    plot_metric_distribution(
        df, "rmsd", "RMSD", "Å",
        figures_dir / "rmsd_distribution.png",
    )
    plot_metric_distribution(
        df, "fident", "Sequence Identity", "fraction",
        figures_dir / "seqid_distribution.png",
    )

    # ---- Step 4: Pairwise heatmaps from foldseek ----
    tm_matrix = build_pairwise_matrix(df, "alntmscore", heatmap_proteins)
    plot_pairwise_heatmap(
        tm_matrix, heatmap_proteins, "TM-score",
        figures_dir / "tmscore_heatmap.png",
        cmap="YlOrRd", vmin=0, vmax=1,
    )

    rmsd_matrix = build_pairwise_matrix(df, "rmsd", heatmap_proteins)
    plot_pairwise_heatmap(
        rmsd_matrix, heatmap_proteins, "RMSD (Å)",
        figures_dir / "rmsd_heatmap.png",
        cmap="viridis_r",
    )

    # ---- Step 5: Position-by-position alignment ----
    plot_position_alignment_heatmap(
        df, figures_dir / "position_alignment_heatmap.png",
        max_proteins=min(100, len(all_proteins)),
    )

    # ---- Step 6: Per-protein residue alignment ----
    plot_residue_alignment_per_protein(
        df, figures_dir / "residue_alignment_per_protein.png",
        max_proteins=50,
    )

    # ---- Step 7: Pairwise Cα DRMSD ----
    if not args.skip_drmsd:
        drmsd_matrix = compute_pairwise_drmsd(pdb_dir, heatmap_proteins)
        plot_pairwise_heatmap(
            drmsd_matrix, heatmap_proteins, "DRMSD (Å)",
            figures_dir / "drmsd_heatmap.png",
            cmap="magma_r",
        )
        # Use DRMSD as the distance matrix for embeddings
        dist_matrix = drmsd_matrix
    else:
        logger.info("Skipping DRMSD computation (use --skip-drmsd to enable).")
        # Fall back to using (1 - TM-score) as a distance proxy for embeddings
        dist_matrix = 1.0 - tm_matrix

    # ---- Step 8: UMAP and MDS embeddings ----
    if len(heatmap_proteins) >= 5:
        plot_embedding(
            dist_matrix, heatmap_proteins, "UMAP",
            figures_dir / "umap_embedding.png",
        )
        plot_embedding(
            dist_matrix, heatmap_proteins, "MDS",
            figures_dir / "mds_embedding.png",
        )
    else:
        logger.warning("Too few proteins (%d) for meaningful embeddings.", len(heatmap_proteins))

    # ---- Step 9: Correlation table ----
    corr = compute_correlation_table(df, output_dir / "correlation_table.csv")
    plot_correlation_heatmap(corr, figures_dir / "correlation_heatmap.png")

    # ---- Summary ----
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  Figures: {figures_dir}/")
    print(f"  Correlation table: {output_dir / 'correlation_table.csv'}")
    print(f"  Foldseek results: {results_path}\n")


if __name__ == "__main__":
    main()
