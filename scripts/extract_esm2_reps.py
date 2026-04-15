"""Extract per-residue ESM-2 embeddings for CATH domains.

Reads PDB files from a directory, extracts sequences, and saves
per-residue representations from the last layer of ESM-2.

Usage:
  python scripts/extract_esm2_reps.py \
      --pdb_dir /path/to/cath_pdbs \
      --save_dir /path/to/output \
      --model_name esm2_t36_3B_UR50D \
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
    """Extract chain sequences from a PDB file using BioPython."""
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
    parser = argparse.ArgumentParser(description="Extract ESM-2 representations for CATH domains")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing CATH PDB files")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for saved representation tensors")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device for inference (default: cuda)")
    parser.add_argument("--model_name", type=str, default="esm2_t36_3B_UR50D",
                        choices=[
                            "esm2_t36_3B_UR50D",
                            "esm2_t33_650M_UR50D",
                            "esm2_t30_150M_UR50D",
                        ],
                        help="ESM-2 model to use (default: esm2_t36_3B_UR50D)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    device = torch.device(args.device)

    # Load model
    print(f"Loading {args.model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"ERROR: No .pdb files found in {pdb_dir}")
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB files")
    total = len(pdb_files)

    for i, pdb_path in enumerate(pdb_files, 1):
        domain_id = pdb_path.stem
        out_path = save_dir / f"{domain_id}_s.pt"

        if out_path.exists():
            print(f"[{i}/{total}] {domain_id} exists, skipping")
            continue

        try:
            chains = extract_sequence_from_pdb(pdb_path)
            if not chains:
                print(f"[{i}/{total}] WARNING: No chains in {pdb_path.name}, skipping")
                continue

            # Concatenate all chains into one sequence
            seq = "".join(chains.values())
            print(f"[{i}/{total}] {domain_id}: {len(seq)} residues")

            batch_labels, batch_strs, batch_tokens = batch_converter(
                [(domain_id, seq)]
            )
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[num_layers],
                    return_contacts=False,
                )

            # Shape: (1, seq_len+2, embed_dim) — +2 for BOS/EOS tokens
            token_reps = results["representations"][num_layers]
            # Strip BOS and EOS -> (seq_len, embed_dim)
            s = token_reps[0, 1:-1, :].cpu()

            torch.save(s, out_path)
            print(f"  Saved: {domain_id}_s.pt shape={list(s.shape)}")

        except Exception as e:
            print(f"[{i}/{total}] ERROR {domain_id}: {e}")

    print(f"\nDone. Processed {total} domains -> {save_dir}")


if __name__ == "__main__":
    main()
