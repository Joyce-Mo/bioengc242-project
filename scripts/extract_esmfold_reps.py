"""Extract per-residue (s) and pairwise (z) representations from ESMFold.

ESMFold (Lin et al., 2023, https://github.com/facebookresearch/esm) runs
ESM-2 as its language model backbone, then feeds the embeddings through an
Evoformer-like folding trunk that produces:
  - s (single representation): per-residue features, shape (L, d_single)
  - z (pair representation):   pairwise features, shape (L, L, d_pair)

These are the representations used for alignment with Protpardelle
(Chu et al., 2024, https://github.com/ProteinDesignLab/protpardelle).

The script uses a forward hook on the folding trunk to capture s and z
before the structure module converts them into 3D coordinates.

Usage:
  python scripts/extract_esmfold_reps.py \
      --pdb_dir /path/to/cath_pdbs \
      --save_dir /path/to/output \
      --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import esm
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def extract_sequence_from_pdb(pdb_path: Path) -> dict[str, str]:
    """Extract chain sequences from a PDB file using BioPython.

    Reads the first model in the PDB and returns a dict mapping chain ID
    to amino acid sequence (one-letter codes). Only standard residues
    (hetflag == ' ') are included.

    Args:
        pdb_path: Path to a PDB file.

    Returns:
        Dictionary of {chain_id: sequence_string}.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    sequences = {}
    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.get_id()[0] == " "]
            if residues:
                seq = "".join(seq1(r.get_resname()) for r in residues)
                sequences[chain.id] = seq
        break  # first model only
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESMFold single (s) and pair (z) representations"
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB files"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Output directory for saved representation tensors"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"],
        help="Device for inference (default: cuda)"
    )
    parser.add_argument(
        "--max_len", type=int, default=1024,
        help=(
            "Skip sequences longer than this to avoid OOM. The pairwise "
            "representation z has shape (L, L, d_pair), so memory grows "
            "quadratically with sequence length. (default: 1024)"
        ),
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    device = torch.device(args.device)

    # Load ESMFold (includes ESM-2 3B backbone + folding trunk)
    # Reference: Lin et al., "Evolutionary-scale prediction of atomic-level
    # protein structure with a language model", Science 2023.
    print("Loading ESMFold v1 (this may take a few minutes)...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)

    # ESMFold can run multiple recycling iterations through the trunk.
    # Setting to 1 is sufficient for representation extraction since we
    # are not using the predicted structure, just the learned features.
    model.set_chunk_size(128)  # gradient checkpointing chunk size for memory

    # Register a forward hook on the folding trunk to capture s and z.
    # The trunk is the Evoformer-like module that transforms the initial
    # single/pair representations into the final s and z used by the
    # structure module.
    captured = {}

    def trunk_hook(module, input, output):
        """Capture the trunk's output dictionary containing s and z."""
        captured["s"] = output["s"].detach().cpu()
        captured["z"] = output["z"].detach().cpu()

    hook_handle = model.trunk.register_forward_hook(trunk_hook)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"ERROR: No .pdb files found in {pdb_dir}")
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB files")
    total = len(pdb_files)
    skipped_exist = 0
    skipped_long = 0
    skipped_error = 0
    processed = 0

    for i, pdb_path in enumerate(pdb_files, 1):
        domain_id = pdb_path.stem
        out_s = save_dir / f"{domain_id}_s.pt"
        out_z = save_dir / f"{domain_id}_z.pt"

        # Skip if both output files already exist
        if out_s.exists() and out_z.exists():
            print(f"[{i}/{total}] {domain_id} exists, skipping")
            skipped_exist += 1
            continue

        try:
            chains = extract_sequence_from_pdb(pdb_path)
            if not chains:
                print(f"[{i}/{total}] WARNING: No chains in {pdb_path.name}, skipping")
                skipped_error += 1
                continue

            # Concatenate all chains into one sequence
            seq = "".join(chains.values())

            if len(seq) > args.max_len:
                print(
                    f"[{i}/{total}] {domain_id}: {len(seq)} residues "
                    f"exceeds max_len={args.max_len}, skipping"
                )
                skipped_long += 1
                continue

            print(f"[{i}/{total}] {domain_id}: {len(seq)} residues")

            # ESMFold's infer() handles tokenization internally and
            # runs the full forward pass (LM + trunk + structure module).
            # Our hook captures s and z from the trunk before coordinates
            # are predicted.
            with torch.no_grad():
                model.infer(seq)

            # Extract captured representations
            # s shape: (1, L, d_single) -> squeeze to (L, d_single)
            # z shape: (1, L, L, d_pair) -> squeeze to (L, L, d_pair)
            s = captured["s"].squeeze(0)
            z = captured["z"].squeeze(0)

            torch.save(s, out_s)
            torch.save(z, out_z)
            print(
                f"  Saved: {domain_id}_s.pt shape={list(s.shape)}, "
                f"{domain_id}_z.pt shape={list(z.shape)}"
            )
            processed += 1

            # Clear captured tensors and GPU cache to free memory
            captured.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[{i}/{total}] ERROR {domain_id}: {e}")
            skipped_error += 1
            captured.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Remove the hook
    hook_handle.remove()

    print(f"\nDone. {processed} processed, {skipped_exist} already existed, "
          f"{skipped_long} skipped (too long), {skipped_error} errors.")
    print(f"Output directory: {save_dir}")


if __name__ == "__main__":
    main()
