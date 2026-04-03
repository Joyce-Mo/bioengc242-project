"""Filter a dataset of protein PDB files.

Applies the following filters to each PDB:
  1. Remove His-tags (runs of >= 6 consecutive HIS residues)
  2. Remove ligands (HETATM records) and water molecules (HOH)
  3. Exclude proteins with > 300 amino acid residues

Filtered PDBs are written to an output directory. Prints summary
statistics comparing the original and filtered datasets.

Usage:
    python scripts/filter_dataset.py [--input-dir PATH] [--output-dir PATH] [--max-residues N]
"""

import argparse
import logging
import sys
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths 
# DEFAULT_INPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20-og"
# DEFAULT_OUTPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20-filtered"
DEFAULT_INPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_converted"
DEFAULT_OUTPUT = "/Users/joycemo/Documents/PhD/Rotation3/dataset/initial_dataset_40/pdb_filtered"
DEFAULT_MAX_RESIDUES = 300

# His-tag detection: minimum consecutive HIS residues to qualify as a tag
# Based on: https://www.thermofisher.com/us/en/home/life-science/antibodies/primary-antibodies/epitope-tag-antibodies/his-tag-antibodies.html 
HIS_TAG_MIN_RUN = 6 


# Helper functions 



def _get_standard_residues(chain):
    """Return list of standard (non-hetero, non-water) residues in a chain.

    Parameters
    ----------
    chain : Bio.PDB.Chain.Chain
        A single chain from a parsed PDB structure.

    Returns
    -------
    list[Bio.PDB.Residue.Residue]
        Residues whose hetero-flag is a blank space (standard ATOM residues).
    """
    return [res for res in chain if res.id[0] == " "]


def _find_his_tag_indices(residues):
    """Identify residue indices belonging to His-tag runs (>= HIS_TAG_MIN_RUN consecutive HIS).

    Parameters
    ----------
    residues : list[Bio.PDB.Residue.Residue]
        Ordered list of standard residues in a chain.

    Returns
    -------
    set[int]
        Indices (into `residues`) that are part of a His-tag run.
    """
    his_tag_idx = set()
    run_start = None
    run_length = 0

    for i, res in enumerate(residues):
        if res.get_resname().strip() == "HIS":
            if run_start is None:
                run_start = i
            run_length += 1
        else:
            # End of a HIS run — mark indices if it was long enough
            if run_length >= HIS_TAG_MIN_RUN:
                his_tag_idx.update(range(run_start, run_start + run_length))
            run_start = None
            run_length = 0

    # Handle run that extends to the end of the chain
    if run_length >= HIS_TAG_MIN_RUN:
        his_tag_idx.update(range(run_start, run_start + run_length))

    return his_tag_idx


class FilterSelect(Select):
    """BioPython PDBIO Select subclass that filters out unwanted residues.

    Removes:
      - Water molecules (resname HOH / WAT)
      - Ligands and other HETATM records (hetero-flag != ' ')
      - His-tag residues (runs of >= HIS_TAG_MIN_RUN consecutive HIS)

    Parameters
    ----------
    his_tag_ids : set[tuple]
        Set of residue full_id tuples to exclude (His-tag residues).
    """

    def __init__(self, his_tag_ids):
        self.his_tag_ids = his_tag_ids

    def accept_residue(self, residue):
        """Accept only standard residues that are not part of a His-tag.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue to evaluate.

        Returns
        -------
        int
            1 to keep, 0 to discard.
        """
        hetero_flag = residue.id[0]

        # Remove water molecules
        if hetero_flag == "W" or residue.get_resname().strip() in ("HOH", "WAT"):
            return 0

        # Remove all other HETATM records (ligands, ions, etc.)
        if hetero_flag != " ":
            return 0

        # Remove His-tag residues
        if residue.get_full_id() in self.his_tag_ids:
            return 0

        return 1


def _count_residues_after_filter(structure, his_tag_ids):
    """Count standard, non-His-tag residues remaining in the first model.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    his_tag_ids : set[tuple]
        Full IDs of His-tag residues to exclude.

    Returns
    -------
    int
        Number of amino acid residues that will remain after filtering.
    """
    count = 0
    model = structure[0]
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if res.get_full_id() in his_tag_ids:
                continue
            count += 1
    return count


 # Main filtering logic

def filter_pdb(pdb_path, output_path, max_residues):
    """Filter a single PDB file.

    Parameters
    ----------
    pdb_path : Path
        Path to the input PDB file.
    output_path : Path
        Path where the filtered PDB will be written.
    max_residues : int
        Maximum number of residues allowed after filtering.

    Returns
    -------
    dict or None
        Statistics dict with keys: 'name', 'original_residues', 'filtered_residues',
        'his_tag_removed', 'hetatm_removed', 'water_removed', 'kept'.
        Returns None if the PDB could not be parsed.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as e:
        logger.warning("Failed to parse %s: %s", pdb_path.name, e)
        return None

    model = structure[0]

    # --- Collect per-file statistics ---
    stats = {
        "name": pdb_path.name,
        "original_residues": 0,
        "filtered_residues": 0,
        "his_tag_removed": 0,
        "hetatm_removed": 0,
        "water_removed": 0,
        "kept": False,
    }

    # Count waters and HETATM (non-water) records
    for chain in model:
        for res in chain:
            hetero_flag = res.id[0]
            if hetero_flag == "W" or res.get_resname().strip() in ("HOH", "WAT"):
                stats["water_removed"] += 1
            elif hetero_flag != " ":
                stats["hetatm_removed"] += 1

    # Identify His-tag residues across all chains
    his_tag_ids = set()
    for chain in model:
        std_residues = _get_standard_residues(chain)
        stats["original_residues"] += len(std_residues)

        tag_indices = _find_his_tag_indices(std_residues)
        stats["his_tag_removed"] += len(tag_indices)
        for idx in tag_indices:
            his_tag_ids.add(std_residues[idx].get_full_id())

    # Count residues remaining after His-tag removal
    filtered_count = _count_residues_after_filter(structure, his_tag_ids)
    stats["filtered_residues"] = filtered_count

    # Apply max-residue cutoff
    if filtered_count > max_residues:
        return stats  # stats["kept"] remains False

    # Write filtered PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), FilterSelect(his_tag_ids))
    stats["kept"] = True

    return stats


def filter_dataset(input_dir, output_dir, max_residues):
    """Filter all PDB files in a directory and write results to output_dir.

    Parameters
    ----------
    input_dir : Path
        Directory containing the original PDB files.
    output_dir : Path
        Directory where filtered PDB files will be written.
    max_residues : int
        Maximum number of residues allowed per protein.

    Returns
    -------
    list[dict]
        List of per-file statistics dicts from filter_pdb().
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        logger.error("No .pdb files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d PDB files in %s", len(pdb_files), input_dir)

    all_stats = []
    for pdb_file in pdb_files:
        out_path = output_dir / pdb_file.name
        result = filter_pdb(pdb_file, out_path, max_residues)
        if result is not None:
            all_stats.append(result)

    return all_stats


def print_summary(all_stats, max_residues):
    """Print a summary of filtering results.

    Parameters
    ----------
    all_stats : list[dict]
        List of per-file statistics dicts.
    max_residues : int
        The max-residue cutoff that was applied.
    """
    total = len(all_stats)
    kept = sum(1 for s in all_stats if s["kept"])
    removed_size = sum(1 for s in all_stats if not s["kept"])
    had_his_tag = sum(1 for s in all_stats if s["his_tag_removed"] > 0)
    had_water = sum(1 for s in all_stats if s["water_removed"] > 0)
    had_hetatm = sum(1 for s in all_stats if s["hetatm_removed"] > 0)

    total_his_removed = sum(s["his_tag_removed"] for s in all_stats)
    total_water_removed = sum(s["water_removed"] for s in all_stats)
    total_hetatm_removed = sum(s["hetatm_removed"] for s in all_stats)

    # Residue length stats for kept proteins
    kept_lengths = [s["filtered_residues"] for s in all_stats if s["kept"]]

    print("\n" + "=" * 60)
    print("DATASET FILTERING SUMMARY")
    print("=" * 60)

    print(f"\n--- Input ---")
    print(f"  Total PDB files processed:       {total}")

    print(f"\n--- Cleaning ---")
    print(f"  Files with His-tags removed:     {had_his_tag}")
    print(f"    Total His-tag residues removed: {total_his_removed}")
    print(f"  Files with water removed:        {had_water}")
    print(f"    Total water molecules removed:  {total_water_removed}")
    print(f"  Files with ligands removed:      {had_hetatm}")
    print(f"    Total HETATM records removed:   {total_hetatm_removed}")

    print(f"\n--- Size filter (max {max_residues} residues) ---")
    print(f"  Removed (too large):             {removed_size}")
    print(f"  Kept:                            {kept}")

    print(f"\n--- Output dataset ---")
    print(f"  Proteins in filtered dataset:    {kept}")
    if kept > 0:
        mean_len = sum(kept_lengths) / len(kept_lengths)
        print(f"  Residue count range:             {min(kept_lengths)} - {max(kept_lengths)}")
        print(f"  Mean residue count:              {mean_len:.1f}")
        print(f"  Median residue count:            {sorted(kept_lengths)[len(kept_lengths) // 2]}")

    print(f"\n  Retention rate:                  {kept / total * 100:.1f}%")
    print("=" * 60 + "\n")


 # CLI
 
def main():
    """Entry point for the dataset filtering script."""
    parser = argparse.ArgumentParser(
        description="Filter protein PDB dataset: remove His-tags, ligands, water, and large proteins.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input directory of PDB files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for filtered PDBs (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=DEFAULT_MAX_RESIDUES,
        help=f"Max residues per protein after filtering (default: {DEFAULT_MAX_RESIDUES})",
    )
    args = parser.parse_args()

    logger.info("Starting dataset filtering...")
    logger.info("  Input:  %s", args.input_dir)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Max residues: %d", args.max_residues)

    all_stats = filter_dataset(args.input_dir, args.output_dir, args.max_residues)
    print_summary(all_stats, args.max_residues)

    logger.info("Done. Filtered PDBs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
