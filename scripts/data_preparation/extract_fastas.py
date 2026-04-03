"""Extract amino acid sequences from protein structure files to FASTA.

Reads all .pdb / .cif / .ent files from an input directory and writes
a single FASTA file with one entry per successfully parsed structure.

Usage:
    python scripts/extract_fastas.py --input-dir PATH --output PATH
"""

import argparse
import logging
import sys
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdb", ".cif", ".ent"}


def _get_parser(file_path):
    ext = file_path.suffix.lower()
    if ext == ".cif":
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def extract_sequence(file_path):
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

    return "".join(str(pp.get_sequence()) for pp in peptides)


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequences from structure files to FASTA.",
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing .pdb / .cif files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output FASTA file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    structure_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not structure_files:
        logger.error("No supported structure files found in %s", input_dir)
        sys.exit(1)

    logger.info("Processing %d structure files from %s", len(structure_files), input_dir)

    written = 0
    failed = 0
    with open(output_path, "w") as fasta:
        for i, f in enumerate(structure_files):
            seq = extract_sequence(f)
            if seq:
                fasta.write(f">{f.name}\n{seq}\n")
                written += 1
            else:
                failed += 1

            if (i + 1) % 10000 == 0:
                logger.info("  Progress: %d / %d files", i + 1, len(structure_files))

    logger.info("Done. Wrote %d sequences to %s (%d failed)", written, output_path, failed)


if __name__ == "__main__":
    main()
