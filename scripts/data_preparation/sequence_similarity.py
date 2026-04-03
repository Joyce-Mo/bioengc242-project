"""Compare sequence similarity between two protein structure datasets.

Extracts amino acid sequences from protein structure files (.pdb, .cif)
in two directories and computes pairwise sequence identity using
BioPython's pairwise aligner. Reports summary statistics and optionally
writes a full similarity matrix to CSV.

Usage:
    python scripts/sequence_similarity.py --dataset-a PATH --dataset-b PATH [--output CSV] [--threshold FLOAT]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Align import PairwiseAligner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


"""
output from seq similarity between cath-20 and karson (filtered) datasets: 

============================================================
SEQUENCE SIMILARITY SUMMARY
============================================================

  Total pairwise comparisons:    288838
  Mean sequence identity:        0.2383 (23.83%)
  Median sequence identity:      0.2444 (24.44%)
  Min sequence identity:         0.0302 (3.02%)
  Max sequence identity:         0.9612 (96.12%)

  Pairs above 30% identity:  34906

  Top 10 most similar pairs:
    132lA00.pdb <-> 7AVG_single_001_density_input.pdb: 0.9612 (96.12%)
    132lA00.pdb <-> 7P6M_single_001_density_input.pdb: 0.9538 (95.38%)
    1a5yA00.pdb <-> 6B8X_single_001_density_input.pdb: 0.9329 (93.29%)
    1ysmA01.pdb <-> 2A26_single_001_density_input.pdb: 0.8542 (85.42%)
    2acfB00.pdb <-> 5SOP_single_001_density_input.pdb: 0.7168 (71.68%)
    2acfB00.pdb <-> 7FRD_single_001_density_input.pdb: 0.7168 (71.68%)
    2i6hA02.pdb <-> 2I6H_single_001_density_input.pdb: 0.5517 (55.17%)
    3zvqA00.pdb <-> 7AVG_single_001_density_input.pdb: 0.5426 (54.26%)
    3zvqA00.pdb <-> 7P6M_single_001_density_input.pdb: 0.5385 (53.85%)
    5tc6A00.pdb <-> 3T94_single_001_density_input.pdb: 0.5165 (51.65%)
============================================================
"""
DEFAULT_DATASET_A = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20-filtered-foldseek"
DEFAULT_DATASET_B = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered"
DEFAULT_OUTPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath_karson_sequence_similarity.csv"
DEFAULT_THRESHOLD = 0.3

SUPPORTED_EXTENSIONS = {".pdb", ".cif", ".ent"}


def _get_parser(file_path):
    """Return the appropriate BioPython parser for the file type.

    Parameters
    ----------
    file_path : Path
        Path to a structure file.

    Returns
    -------
    PDBParser or MMCIFParser
    """
    ext = file_path.suffix.lower()
    if ext == ".cif":
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def extract_sequence(file_path):
    """Extract the amino acid sequence from a protein structure file.

    Concatenates sequences from all polypeptide chains in the first model.

    Parameters
    ----------
    file_path : Path
        Path to a .pdb or .cif file.

    Returns
    -------
    str or None
        One-letter amino acid sequence, or None on failure.
    """
    parser = _get_parser(file_path)
    try:
        structure = parser.get_structure(file_path.stem, str(file_path))
    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path.name, e)
        return None

    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure[0])

    if not peptides:
        logger.warning("No polypeptides found in %s", file_path.name)
        return None

    sequence = "".join(str(pp.get_sequence()) for pp in peptides)
    return sequence


def load_dataset(directory):
    """Load all protein sequences from structure files in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing .pdb / .cif files.

    Returns
    -------
    dict[str, str]
        Mapping of filename -> amino acid sequence.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error("Directory does not exist: %s", directory)
        sys.exit(1)

    sequences = {}
    structure_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not structure_files:
        logger.error("No supported structure files found in %s", directory)
        sys.exit(1)

    logger.info("Loading %d structure files from %s", len(structure_files), directory)

    for f in structure_files:
        seq = extract_sequence(f)
        if seq:
            sequences[f.name] = seq

    logger.info("Successfully extracted %d sequences", len(sequences))
    return sequences


def compute_sequence_identity(seq_a, seq_b):
    """Compute pairwise sequence identity between two sequences.

    Uses a global alignment and reports the fraction of identical
    residues relative to the length of the longer sequence.

    Parameters
    ----------
    seq_a : str
        First amino acid sequence.
    seq_b : str
        Second amino acid sequence.

    Returns
    -------
    float
        Sequence identity as a fraction in [0, 1].
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    alignments = aligner.align(seq_a, seq_b)
    best = alignments[0]

    # Count identical positions
    aligned_a, aligned_b = best[0], best[1]
    matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b and a != "-")

    max_len = max(len(seq_a), len(seq_b))
    return matches / max_len if max_len > 0 else 0.0


def compare_datasets(seqs_a, seqs_b):
    """Compute all pairwise sequence identities between two datasets.

    Parameters
    ----------
    seqs_a : dict[str, str]
        Sequences from dataset A.
    seqs_b : dict[str, str]
        Sequences from dataset B.

    Returns
    -------
    list[dict]
        List of dicts with keys: 'name_a', 'name_b', 'identity', 'len_a', 'len_b'.
    """
    results = []
    total_pairs = len(seqs_a) * len(seqs_b)
    logger.info("Computing %d pairwise comparisons...", total_pairs)

    done = 0
    for name_a, seq_a in seqs_a.items():
        for name_b, seq_b in seqs_b.items():
            identity = compute_sequence_identity(seq_a, seq_b)
            results.append({
                "name_a": name_a,
                "name_b": name_b,
                "identity": identity,
                "len_a": len(seq_a),
                "len_b": len(seq_b),
            })
            done += 1
            if done % 500 == 0:
                logger.info("  Progress: %d / %d (%.1f%%)", done, total_pairs, done / total_pairs * 100)

    return results


def print_summary(results, threshold):
    """Print summary statistics of pairwise comparisons.

    Parameters
    ----------
    results : list[dict]
        Pairwise comparison results.
    threshold : float
        Identity threshold for counting "similar" pairs.
    """
    if not results:
        print("No comparisons to summarise.")
        return

    identities = [r["identity"] for r in results]
    above_threshold = [r for r in results if r["identity"] >= threshold]

    mean_id = sum(identities) / len(identities)
    sorted_ids = sorted(identities)
    median_id = sorted_ids[len(sorted_ids) // 2]

    print("\n" + "=" * 60)
    print("SEQUENCE SIMILARITY SUMMARY")
    print("=" * 60)
    print(f"\n  Total pairwise comparisons:    {len(results)}")
    print(f"  Mean sequence identity:        {mean_id:.4f} ({mean_id * 100:.2f}%)")
    print(f"  Median sequence identity:      {median_id:.4f} ({median_id * 100:.2f}%)")
    print(f"  Min sequence identity:         {sorted_ids[0]:.4f} ({sorted_ids[0] * 100:.2f}%)")
    print(f"  Max sequence identity:         {sorted_ids[-1]:.4f} ({sorted_ids[-1] * 100:.2f}%)")
    print(f"\n  Pairs above {threshold:.0%} identity:  {len(above_threshold)}")

    if above_threshold:
        print(f"\n  Top 10 most similar pairs:")
        top = sorted(above_threshold, key=lambda r: r["identity"], reverse=True)[:10]
        for r in top:
            print(f"    {r['name_a']} <-> {r['name_b']}: {r['identity']:.4f} ({r['identity'] * 100:.2f}%)")

    print("=" * 60 + "\n")


def write_results_csv(results, output_path):
    """Write pairwise results to a CSV file.

    Parameters
    ----------
    results : list[dict]
        Pairwise comparison results.
    output_path : Path
        Path to the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name_a", "name_b", "identity", "len_a", "len_b"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results written to %s", output_path)


def main():
    """Entry point for sequence similarity comparison."""
    parser = argparse.ArgumentParser(
        description="Compare sequence similarity between two protein structure datasets.",
    )
    parser.add_argument(
        "--dataset-a",
        type=str,
        default=DEFAULT_DATASET_A,
        help=f"Directory containing structure files for dataset A (default: {DEFAULT_DATASET_A})",
    )
    parser.add_argument(
        "--dataset-b",
        type=str,
        default=DEFAULT_DATASET_B,
        help=f"Directory containing structure files for dataset B (default: {DEFAULT_DATASET_B})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to write full pairwise results CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    args = parser.parse_args()

    logger.info("Dataset A: %s", args.dataset_a)
    logger.info("Dataset B: %s", args.dataset_b)

    seqs_a = load_dataset(args.dataset_a)
    seqs_b = load_dataset(args.dataset_b)

    results = compare_datasets(seqs_a, seqs_b)
    print_summary(results, args.threshold)

    if args.output:
        write_results_csv(results, args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
