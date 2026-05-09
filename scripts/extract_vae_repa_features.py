#!/usr/bin/env python
"""Extract per-residue VAE features for REPA alignment with Protpardelle.

For each PDB in the training set, computes the 7-channel (C, L, L) feature
map via featurize_pdb.py, then extracts a per-residue representation of
shape [L, D] that can be used as expert features in Protpardelle's REPA
(Representation Alignment) framework.

Per-residue feature extraction from the (7, L, L) pair feature map:
    - Row-wise slice: for residue i, row i of each channel gives that
      residue's pairwise relationship with all other residues. Shape: (7, L).
    - Diagonal: self-interaction values (7 scalars per residue).
    - Row mean: average pairwise signal (7 scalars per residue).
    - Row max: strongest pairwise interaction (7 scalars per residue).

The default mode ("row") saves the full row-wise features [L, 7*L] which
preserves all pairwise information. The "pooled" mode saves a compact
[L, 21] vector (diagonal + row mean + row max) that is cheaper to store.

Optionally, the features can be passed through the frozen VAE encoder
to extract intermediate convolutional representations ("encoder" mode).

Output format matches protpardelle's REPA convention:
    {output_dir}/{pdb_stem}.pt  -- torch tensor of shape [L, D]

Usage:
    # Full row features (D = 7*L, use for small datasets)
    python scripts/extract_vae_repa_features.py \
        --pdb-dir /path/to/training_pdbs \
        --vae-checkpoint /path/to/vae_best.pt \
        --output-dir /path/to/vae_repa_features \
        --mode pooled

    # On Anvil (SLURM array for large dataset):
    python scripts/extract_vae_repa_features.py \
        --pdb-dir /anvil/scratch/x-jmo/datasets/augmented_ingraham_cath_bugfree \
        --vae-checkpoint /path/to/vae_best.pt \
        --output-dir /anvil/scratch/x-jmo/datasets/vae_repa_features \
        --mode pooled \
        --task-id $SLURM_ARRAY_TASK_ID \
        --n-tasks $SLURM_ARRAY_TASK_COUNT

References:
    - REPA: Yu et al. "Representation Alignment for Generation: Training
      Diffusion Transformers Is Easier Than You Think" (2024).
    - iREPA: Teng & Lin, "Improved Representation Alignment for Generation"
    - Protpardelle REPA: protpardelle-1c/src/protpardelle/core/modules.py
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Constants (must match vae.py / featurize_pdb.py)
CROP_SIZE = 64
N_CHANNELS = 7


def load_featurizer():
    """Import featurize_pdb module."""
    import importlib.util
    fp = Path(__file__).resolve().parent.parent / "vae" / "featurize_pdb.py"
    spec = importlib.util.spec_from_file_location("featurize_pdb", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_pooled_features(feat_map):
    """Extract per-residue features via diagonal + row pooling.

    Args:
        feat_map: (C, L, L) numpy array

    Returns:
        (L, 3*C) numpy array -- compact per-residue representation
        For C=7: D=21 per residue.
    """
    C, L, _ = feat_map.shape

    # Diagonal: (C, L) -> (L, C)
    diag = np.array([np.diag(feat_map[c]) for c in range(C)]).T

    # Row-wise mean: (C, L) -> (L, C)
    row_mean = feat_map.mean(axis=2).T

    # Row-wise max: (C, L) -> (L, C)
    row_max = feat_map.max(axis=2).T

    # Concatenate: (L, 3*C)
    return np.concatenate([diag, row_mean, row_max], axis=1).astype(np.float32)


def extract_row_features(feat_map):
    """Extract per-residue features as full row slices.

    For residue i, takes row i from all channels and concatenates.

    Args:
        feat_map: (C, L, L) numpy array

    Returns:
        (L, C*L) numpy array -- full pairwise information per residue
    """
    C, L, _ = feat_map.shape
    # feat_map is (C, L, L). For each residue i, row i gives (C, L).
    # Reshape to (L, C*L)
    return feat_map.transpose(1, 0, 2).reshape(L, C * L).astype(np.float32)


def extract_encoder_features(feat_map, vae_model, device="cpu"):
    """Extract per-residue features from VAE encoder intermediate layers.

    Runs the feature map through the VAE encoder and captures activations
    after the first two conv blocks (before spatial resolution drops too low).
    The first conv block outputs (16, 32, 32) -- we average-pool the spatial
    dim back to L to get per-residue features.

    Args:
        feat_map: (C, L, L) numpy array
        vae_model: loaded VAE model (frozen)
        device: torch device

    Returns:
        (L, D) numpy array where D = 16 (channels from conv1)
    """
    L = feat_map.shape[1]

    # Pad/crop to CROP_SIZE for the VAE
    if L >= CROP_SIZE:
        fm = feat_map[:, :CROP_SIZE, :CROP_SIZE]
        out_L = CROP_SIZE
    else:
        pad = CROP_SIZE - L
        fm = np.pad(feat_map, ((0, 0), (0, pad), (0, pad)), mode="constant")
        out_L = L

    x = torch.from_numpy(fm).unsqueeze(0).to(device)  # (1, C, 64, 64)

    # Hook into conv1 output (16 channels, 32x32)
    activations = {}
    def hook_fn(module, input, output):
        activations["conv1"] = output.detach()

    handle = vae_model.conv1.register_forward_hook(hook_fn)
    with torch.no_grad():
        vae_model.encode(x)
    handle.remove()

    # activations["conv1"] is (1, 16, 32, 32)
    act = activations["conv1"].squeeze(0).cpu().numpy()  # (16, 32, 32)

    # Upsample back to L residues: take diagonal-adjacent features
    # Since conv1 has stride 2, position j in the 32x32 map corresponds
    # to residues 2j and 2j+1. We interpolate back to L.
    from scipy.ndimage import zoom
    scale = out_L / act.shape[1]
    # Zoom spatial dims back to L
    act_upsampled = zoom(act, (1, scale, scale), order=1)  # (16, L, L)

    # Extract per-residue: diagonal of each channel
    per_res = np.array([np.diag(act_upsampled[c, :out_L, :out_L])
                        for c in range(act_upsampled.shape[0])]).T  # (L, 16)

    return per_res[:out_L].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-residue VAE features for Protpardelle REPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdb-dir", type=str, required=True,
                        help="Directory of training PDB files")
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="Path to vae_best.pt (required for 'encoder' mode)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for {pdb_stem}.pt files")
    parser.add_argument("--mode", choices=["pooled", "row", "encoder"],
                        default="pooled",
                        help="Feature extraction mode (default: pooled). "
                             "'pooled' gives D=21, 'row' gives D=7*L, "
                             "'encoder' gives D=16 from VAE conv1")
    parser.add_argument("--task-id", type=int, default=None,
                        help="1-indexed SLURM array task ID")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Total number of SLURM array tasks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract features that already exist")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Gather PDB paths
    pdb_paths = sorted(Path(args.pdb_dir).rglob("*.pdb"))
    if not pdb_paths:
        sys.exit(f"No PDB files found under {args.pdb_dir}")

    # SLURM array chunking
    if args.task_id is not None and args.n_tasks is not None:
        chunk_size = math.ceil(len(pdb_paths) / args.n_tasks)
        start = (args.task_id - 1) * chunk_size
        end = min(start + chunk_size, len(pdb_paths))
        if start >= len(pdb_paths):
            print(f"Task {args.task_id} has nothing to do ({len(pdb_paths)} PDBs / {args.n_tasks} tasks)")
            return
        pdb_paths = pdb_paths[start:end]

    print(f"Extracting {args.mode} features for {len(pdb_paths)} PDBs -> {outdir}")

    # Load featurizer
    fpdb = load_featurizer()

    # Load VAE if needed
    vae_model = None
    if args.mode == "encoder":
        if args.vae_checkpoint is None:
            sys.exit("--vae-checkpoint required for encoder mode")
        from vae.structure_module import load_vae
        vae_model, _ = load_vae(args.vae_checkpoint, device=args.device)

    n_ok, n_skip, n_fail = 0, 0, 0
    manifest = {}

    for i, pdb_path in enumerate(pdb_paths):
        stem = pdb_path.stem
        out_path = outdir / f"{stem}.pt"

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            feat_map = fpdb.featurize_pdb(str(pdb_path))
        except Exception as e:
            print(f"  FAIL {stem}: {e}", file=sys.stderr)
            n_fail += 1
            continue

        L = feat_map.shape[1]  # original protein length (before crop/pad)

        if args.mode == "pooled":
            per_res = extract_pooled_features(feat_map)
        elif args.mode == "row":
            per_res = extract_row_features(feat_map)
        elif args.mode == "encoder":
            per_res = extract_encoder_features(feat_map, vae_model, device=args.device)

        # Save as [L, D] tensor (no crop/pad -- use original protein length)
        tensor = torch.from_numpy(per_res[:L])
        torch.save(tensor, out_path)
        manifest[stem] = L
        n_ok += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pdb_paths)}: {n_ok} ok, {n_skip} skip, {n_fail} fail")

    # Save manifest
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    D = per_res.shape[1] if n_ok > 0 else "?"
    print(f"\nDone: {n_ok} written, {n_skip} skipped, {n_fail} failed")
    print(f"Feature dim: D={D}")
    print(f"Set expert_dim={D} in your REPA yaml config")


if __name__ == "__main__":
    main()
