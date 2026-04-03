"""Filter a protein PDB dataset by foldseek TM-score.

Runs foldseek all-vs-all structural alignment, then keeps only proteins
that have at least one non-self alignment with TM-score >= a threshold
(default 0.5). This removes structurally isolated proteins that share
no significant fold similarity with anything else in the dataset.

Outputs:
  - Filtered PDB files copied to the output directory
  - foldseek_results.tsv with all pairwise alignments
  - Summary statistics printed to stdout

Usage:
    python scripts/foldseek_tm_filter.py
    python scripts/foldseek_tm_filter.py --input-dir /path/to/pdbs --output-dir /path/to/output
    python scripts/foldseek_tm_filter.py --tm-threshold 0.6 --threads 8
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

  # Default paths
DEFAULT_INPUT_DIR = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered"
DEFAULT_OUTPUT_DIR = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered_foldseek_filtered"
DEFAULT_TM_THRESHOLD = 0.5

# Fields requested from foldseek easy-search
# Must match FOLDSEEK_COLUMNS in foldseek_analysis.py so the TSV can be
# reused directly for visualization.
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


  # Step 1: Run foldseek
  
def run_foldseek(input_dir, work_dir, tm_threshold, threads=4):
    """Run foldseek easy-search in all-vs-all mode.

    Parameters
    ----------
    input_dir : Path
        Directory containing PDB files.
    work_dir : Path
        Working directory for foldseek output and temp files.
    tm_threshold : float
        Minimum TM-score for foldseek to report an alignment.
    threads : int
        Number of threads for foldseek.

    Returns
    -------
    Path
        Path to the foldseek results TSV file.
    """
    results_path = work_dir / "foldseek_results.tsv"
    tmp_dir = work_dir / "foldseek_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        str(input_dir),
        str(input_dir),
        str(results_path),
        str(tmp_dir),
        "--format-output", FOLDSEEK_FORMAT_STR,
        "--threads", str(threads),
        "--exhaustive-search",
        "--tmscore-threshold", str(tm_threshold),
    ]

    logger.info("Running foldseek: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Foldseek stderr:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Foldseek finished. Results at %s", results_path)
    return results_path


  # Step 2: Filter by TM-score
  
def filter_by_tmscore(results_path, tm_threshold):
    """Identify proteins that have at least one non-self hit above the TM threshold.

    Parameters
    ----------
    results_path : Path
        Path to foldseek results TSV.
    tm_threshold : float
        Minimum TM-score for a non-self alignment to count.

    Returns
    -------
    tuple[set[str], pd.DataFrame]
        (set of protein names to keep, full results DataFrame).
    """
    df = pd.read_csv(results_path, sep="\t", header=None, names=FOLDSEEK_COLUMNS)

    # Clean protein names (strip .pdb extension)
    df["query"] = df["query"].astype(str).str.replace(r"\.pdb$", "", regex=True)
    df["target"] = df["target"].astype(str).str.replace(r"\.pdb$", "", regex=True)

    logger.info("Loaded %d total alignments between %d unique proteins",
                len(df), df["query"].nunique())

    # Non-self alignments above TM threshold
    hits = df[(df["query"] != df["target"]) & (df["alntmscore"] >= tm_threshold)]
    logger.info("Non-self alignments with TM-score >= %.2f: %d", tm_threshold, len(hits))

    # Proteins that appear as query OR target in a qualifying hit
    kept_proteins = set(hits["query"].unique()) | set(hits["target"].unique())

    return kept_proteins, df


  # Step 3: Copy kept PDBs
  
def copy_kept_pdbs(input_dir, output_dir, kept_proteins):
    """Copy PDB files for kept proteins to the output directory.

    Parameters
    ----------
    input_dir : Path
        Source directory of PDB files.
    output_dir : Path
        Destination directory for filtered PDB files.
    kept_proteins : set[str]
        Set of protein names (without .pdb) to keep.

    Returns
    -------
    tuple[int, int]
        (number of files copied, number of input files total).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    input_pdbs = sorted(input_dir.glob("*.pdb"))
    total = len(input_pdbs)
    copied = 0

    for pdb_file in input_pdbs:
        stem = pdb_file.stem
        if stem in kept_proteins:
            shutil.copy2(pdb_file, output_dir / pdb_file.name)
            copied += 1

    return copied, total


  # Step 4: Print summary stats
  
def print_summary(df, kept_proteins, all_proteins, tm_threshold, copied, total):
    """Print filtering summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Full foldseek results DataFrame.
    kept_proteins : set[str]
        Protein names that passed the TM filter.
    all_proteins : set[str]
        All protein names in the input dataset.
    tm_threshold : float
        The TM-score cutoff used.
    copied : int
        Number of PDB files written to output.
    total : int
        Total number of input PDB files.
    """
    removed_proteins = all_proteins - kept_proteins

    # TM-score stats for non-self hits
    non_self = df[df["query"] != df["target"]]
    tm_scores = non_self["alntmscore"]

    # Per-protein max TM-score (best structural match)
    max_tm_per_protein = non_self.groupby("query")["alntmscore"].max()

    # Stats for kept proteins
    kept_non_self = non_self[
        non_self["query"].isin(kept_proteins) & non_self["target"].isin(kept_proteins)
    ]
    kept_tm = kept_non_self["alntmscore"] if not kept_non_self.empty else pd.Series(dtype=float)

    print("\n" + "=" * 60)
    print("FOLDSEEK TM-SCORE FILTERING SUMMARY")
    print("=" * 60)

    print(f"\n--- Input ---")
    print(f"  Total PDB files:                 {total}")
    print(f"  Proteins in foldseek results:    {len(all_proteins)}")
    print(f"  Total pairwise alignments:       {len(df)}")
    print(f"  Non-self alignments:             {len(non_self)}")

    print(f"\n--- TM-score statistics (all non-self) ---")
    if not tm_scores.empty:
        print(f"  Mean TM-score:                   {tm_scores.mean():.3f}")
        print(f"  Median TM-score:                 {tm_scores.median():.3f}")
        print(f"  Min TM-score:                    {tm_scores.min():.3f}")
        print(f"  Max TM-score:                    {tm_scores.max():.3f}")

    print(f"\n--- Per-protein best TM-score ---")
    if not max_tm_per_protein.empty:
        print(f"  Mean best TM-score:              {max_tm_per_protein.mean():.3f}")
        print(f"  Proteins with best TM < {tm_threshold:.1f}:     "
              f"{(max_tm_per_protein < tm_threshold).sum()}")

    print(f"\n--- Filtering (TM threshold = {tm_threshold}) ---")
    print(f"  Proteins kept:                   {len(kept_proteins)}")
    print(f"  Proteins removed:                {len(removed_proteins)}")
    print(f"  PDB files copied to output:      {copied}")

    print(f"\n--- Output dataset ---")
    print(f"  Proteins in filtered dataset:    {copied}")
    if not kept_tm.empty:
        print(f"  Mean TM-score (kept pairs):      {kept_tm.mean():.3f}")
        print(f"  Median TM-score (kept pairs):    {kept_tm.median():.3f}")
    print(f"  Retention rate:                  {copied / max(total, 1) * 100:.1f}%")

    if removed_proteins and len(removed_proteins) <= 20:
        print(f"\n--- Removed proteins ---")
        for p in sorted(removed_proteins):
            best = max_tm_per_protein.get(p, 0.0)
            print(f"  {p}  (best TM = {best:.3f})")

    print("=" * 60 + "\n")


  # CLI
  
def main():
    """Entry point for foldseek TM-score filtering."""
    parser = argparse.ArgumentParser(
        description="Filter protein dataset by foldseek TM-score.",
    )
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_INPUT_DIR,
        help=f"Input directory of PDB files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for filtered PDBs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--tm-threshold", type=float, default=DEFAULT_TM_THRESHOLD,
        help=f"Min TM-score for a protein to be kept (default: {DEFAULT_TM_THRESHOLD}).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of threads for foldseek (default: 4).",
    )
    parser.add_argument(
        "--skip-foldseek", action="store_true",
        help="Skip foldseek run and reuse existing results TSV in output-dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Step 1: Run foldseek (or reuse existing results)
    results_path = output_dir / "foldseek_results.tsv"
    if args.skip_foldseek and results_path.exists():
        logger.info("Skipping foldseek, reusing: %s", results_path)
    else:
        results_path = run_foldseek(input_dir, output_dir, args.tm_threshold, args.threads)

    # Step 2: Filter by TM-score
    kept_proteins, df = filter_by_tmscore(results_path, args.tm_threshold)

    # All proteins in input directory
    all_input_proteins = {f.stem for f in input_dir.glob("*.pdb")}

    # Step 3: Copy kept PDBs to output
    copied, total = copy_kept_pdbs(input_dir, output_dir, kept_proteins)

    # Step 4: Print summary
    all_foldseek_proteins = set(df["query"].unique()) | set(df["target"].unique())
    print_summary(df, kept_proteins, all_foldseek_proteins,
                  args.tm_threshold, copied, total)

    logger.info("Done. Filtered PDBs written to %s", output_dir)


if __name__ == "__main__":
    main()
