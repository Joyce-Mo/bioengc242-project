"""Compare sequence similarity between two protein datasets using MMseqs2.

Accepts either pre-extracted FASTA files (--fasta-a / --fasta-b) or
directories of structure files (--dataset-a / --dataset-b). When given
directories, extracts sequences with BioPython first. Then runs MMseqs2
easy-search for fast all-vs-all sequence comparison.

Requires: mmseqs2 (conda install -c conda-forge -c bioconda mmseqs2)
          biopython (only needed if using --dataset-a / --dataset-b)

Usage:
    # With pre-extracted FASTAs (fast):
    python scripts/sequence_similarity_mmseqs2.py --fasta-a A.fasta --fasta-b B.fasta --output out.csv

    # With structure directories (slow for large datasets):
    python scripts/sequence_similarity_mmseqs2.py --dataset-a DIR_A --dataset-b DIR_B --output out.csv
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_A = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20-filtered-foldseek"
DEFAULT_DATASET_B = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered"
DEFAULT_OUTPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath_karson_sequence_similarity_mmseqs2.csv"
DEFAULT_THRESHOLD = 0.3

SUPPORTED_EXTENSIONS = {".pdb", ".cif", ".ent"}


def _get_parser(file_path):
    """Return the appropriate BioPython parser for the file type."""
    ext = file_path.suffix.lower()
    if ext == ".cif":
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def extract_sequence(file_path):
    """Extract the amino acid sequence from a protein structure file."""
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


def write_fasta(sequences, fasta_path):
    """Write sequences dict to a FASTA file."""
    with open(fasta_path, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")
    logger.info("Wrote %d sequences to %s", len(sequences), fasta_path)


def run_mmseqs2_search(fasta_a, fasta_b, output_tsv, tmp_dir, min_seq_id=0.0):
    """Run MMseqs2 easy-search between two FASTA files."""
    cmd = [
        "mmseqs", "easy-search",
        fasta_a,
        fasta_b,
        output_tsv,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "--format-output", "query,target,fident,alnlen,qlen,tlen,evalue,bits",
        "-s", "7.5",
        "--threads", str(os.cpu_count() or 4),
        "--max-seqs", "300",
    ]

    logger.info("Running MMseqs2: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("MMseqs2 failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("MMseqs2 search complete")


def parse_mmseqs2_results(output_tsv):
    """Parse MMseqs2 easy-search output into a list of result dicts."""
    results = []
    with open(output_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            query, target, fident, alnlen, qlen, tlen, evalue, bits = parts
            results.append({
                "name_a": query,
                "name_b": target,
                "identity": float(fident),
                "len_a": int(qlen),
                "len_b": int(tlen),
            })

    logger.info("Parsed %d hits from MMseqs2 output", len(results))
    return results


def print_summary(results, threshold):
    """Print summary statistics of pairwise comparisons."""
    if not results:
        print("No comparisons to summarise.")
        return

    identities = [r["identity"] for r in results]
    above_threshold = [r for r in results if r["identity"] >= threshold]

    mean_id = sum(identities) / len(identities)
    sorted_ids = sorted(identities)
    median_id = sorted_ids[len(sorted_ids) // 2]

    print("\n" + "=" * 60)
    print("SEQUENCE SIMILARITY SUMMARY (MMseqs2)")
    print("=" * 60)
    print(f"\n  Total hits reported:           {len(results)}")
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
    """Write pairwise results to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name_a", "name_b", "identity", "len_a", "len_b"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results written to %s", output_path)


def main():
    """Entry point for MMseqs2-based sequence similarity comparison."""
    parser = argparse.ArgumentParser(
        description="Compare sequence similarity between two protein structure datasets using MMseqs2.",
    )
    # Input: either FASTA files or structure directories
    input_group = parser.add_argument_group("Input (use FASTA files OR structure directories)")
    input_group.add_argument(
        "--fasta-a",
        type=str,
        default=None,
        help="Pre-extracted FASTA file for dataset A (skips PDB parsing)",
    )
    input_group.add_argument(
        "--fasta-b",
        type=str,
        default=None,
        help="Pre-extracted FASTA file for dataset B (skips PDB parsing)",
    )
    input_group.add_argument(
        "--dataset-a",
        type=str,
        default=DEFAULT_DATASET_A,
        help=f"Directory containing structure files for dataset A (default: {DEFAULT_DATASET_A})",
    )
    input_group.add_argument(
        "--dataset-b",
        type=str,
        default=DEFAULT_DATASET_B,
        help=f"Directory containing structure files for dataset B (default: {DEFAULT_DATASET_B})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to write results CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=0.0,
        help="Minimum sequence identity for MMseqs2 to report a hit (default: 0.0 = all hits)",
    )
    args = parser.parse_args()

    if shutil.which("mmseqs") is None:
        logger.error("mmseqs2 not found. Install with: conda install -c conda-forge -c bioconda mmseqs2")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Resolve FASTA inputs: use pre-extracted or extract from directories
        if args.fasta_a:
            fasta_a = args.fasta_a
            logger.info("Using pre-extracted FASTA A: %s", fasta_a)
        else:
            logger.info("Extracting sequences from directory A: %s", args.dataset_a)
            seqs_a = load_dataset(args.dataset_a)
            fasta_a = os.path.join(tmp_dir, "dataset_a.fasta")
            write_fasta(seqs_a, fasta_a)

        if args.fasta_b:
            fasta_b = args.fasta_b
            logger.info("Using pre-extracted FASTA B: %s", fasta_b)
        else:
            logger.info("Extracting sequences from directory B: %s", args.dataset_b)
            seqs_b = load_dataset(args.dataset_b)
            fasta_b = os.path.join(tmp_dir, "dataset_b.fasta")
            write_fasta(seqs_b, fasta_b)

        mmseqs_out = os.path.join(tmp_dir, "mmseqs_results.tsv")
        mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")

        run_mmseqs2_search(fasta_a, fasta_b, mmseqs_out, mmseqs_tmp,
                           min_seq_id=args.min_seq_id)

        results = parse_mmseqs2_results(mmseqs_out)

    print_summary(results, args.threshold)

    if args.output:
        write_results_csv(results, args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
