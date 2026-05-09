#!/usr/bin/env python
"""Compare VAE and diffusion model (Protpardelle) in the VAE feature space.

Uses the production VAE encoder as a shared evaluation tool:
1. Featurize reference CATH PDBs and protpardelle-generated PDBs with the
   same 7-channel pipeline (featurize_pdb.py)
2. Encode all feature maps through the frozen VAE encoder to get latent mu
3. UMAP of the latent space: CATH reference vs protpardelle vs VAE decoded
4. Feature-level comparison: distance/contact distributions across methods

This directly compares both generative models in the same representation
space without retraining anything.

Usage:
    python scripts/evaluation/compare_vae_diffusion.py \
        --vae-checkpoint /path/to/vae_best.pt \
        --cath-dir /path/to/cath_pdbs \
        --diffusion-dir /path/to/protpardelle/results/cc89_variant_samples \
        --outdir output/vae_vs_diffusion
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Constants
CROP_SIZE = 64
N_CHANNELS = 7
D_MAX = 22.0
CHANNEL_NAMES = [
    "CA-CA dist", "CB-CB dist", "Contact",
    "Hydrophobicity", "Charge", "Polarity", "SASA",
]


def load_featurizer():
    import importlib.util
    fp = Path(__file__).resolve().parent.parent.parent / "vae" / "featurize_pdb.py"
    spec = importlib.util.spec_from_file_location("featurize_pdb", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vae(checkpoint_path, device="cpu"):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from vae.structure_module import VAE
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hparams", {})
    z_dim = ckpt.get("z_dim", hparams.get("z_dim", 64))
    model = VAE(
        in_channels=ckpt.get("in_channels", N_CHANNELS),
        z_dim=z_dim,
        dropout=hparams.get("dropout", 0.0),
        use_batchnorm=hparams.get("use_batchnorm", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, z_dim


def featurize_pdbs(pdb_dir, fpdb, max_files=None, label=""):
    """Featurize PDBs and return (features, paths) where features are cropped to CROP_SIZE."""
    pdb_files = sorted(Path(pdb_dir).rglob("*.pdb"))
    if max_files:
        pdb_files = pdb_files[:max_files]
    print(f"  [{label}] Featurizing {len(pdb_files)} PDBs from {pdb_dir}")

    features, stems = [], []
    for i, pdb_file in enumerate(pdb_files):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(pdb_files)}...")
        try:
            feats = fpdb.featurize_pdb(str(pdb_file))
        except Exception:
            continue
        _, h, _ = feats.shape
        if h >= CROP_SIZE:
            feats = feats[:, :CROP_SIZE, :CROP_SIZE]
        else:
            pad = CROP_SIZE - h
            feats = np.pad(feats, ((0, 0), (0, pad), (0, pad)), mode="constant")
        features.append(feats)
        stems.append(pdb_file.stem)

    print(f"    {len(features)} featurized successfully")
    return np.stack(features), stems


def encode_features(vae, features_np, device="cpu", batch_size=64):
    """Encode feature maps through VAE encoder to get latent mu vectors."""
    all_mu = []
    n = features_np.shape[0]
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features_np[i:i+batch_size]).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(batch)
        all_mu.append(mu.cpu().numpy())
    return np.concatenate(all_mu, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compare VAE and diffusion model")
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--cath-dir", type=str, required=True,
                        help="Reference CATH PDB directory")
    parser.add_argument("--diffusion-dir", type=str, required=True,
                        help="Protpardelle sample directory (with PDB files)")
    parser.add_argument("--cath-max", type=int, default=300)
    parser.add_argument("--diffusion-max", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="output/vae_vs_diffusion")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae, z_dim = load_vae(args.vae_checkpoint, device=args.device)

    # Load featurizer
    fpdb = load_featurizer()

    # Featurize all sources
    print("\nFeaturizing PDBs...")
    cath_feats, cath_stems = featurize_pdbs(
        args.cath_dir, fpdb, max_files=args.cath_max, label="CATH"
    )
    diff_feats, diff_stems = featurize_pdbs(
        args.diffusion_dir, fpdb, max_files=args.diffusion_max, label="Diffusion"
    )

    # Generate VAE samples for comparison
    print("\nGenerating VAE samples...")
    n_vae = min(len(diff_stems), 50)  # match diffusion count
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(n_vae, z_dim, device=args.device)
        vae_decoded = vae.decode(z).cpu().numpy()
    print(f"  Generated {n_vae} VAE samples")

    # Encode all through VAE encoder to get latent embeddings
    print("\nEncoding through VAE encoder...")
    cath_mu = encode_features(vae, cath_feats, device=args.device)
    diff_mu = encode_features(vae, diff_feats, device=args.device)
    vae_mu = encode_features(vae, vae_decoded, device=args.device)
    print(f"  CATH: {cath_mu.shape}, Diffusion: {diff_mu.shape}, VAE: {vae_mu.shape}")

    # Figure 1: UMAP of latent space
    print("\nComputing UMAP...")
    from sklearn.manifold import TSNE
    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("  umap-learn not installed, falling back to t-SNE")

    all_mu = np.concatenate([cath_mu, diff_mu, vae_mu], axis=0)
    labels = (
        ["CATH reference"] * len(cath_mu)
        + ["Protpardelle (diffusion)"] * len(diff_mu)
        + ["VAE decoded"] * len(vae_mu)
    )

    if has_umap:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed)
        embedding = reducer.fit_transform(all_mu)
        method_name = "UMAP"
    else:
        reducer = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(all_mu)-1))
        embedding = reducer.fit_transform(all_mu)
        method_name = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"CATH reference": "#999999", "Protpardelle (diffusion)": "#E07B54", "VAE decoded": "#5B8DB8"}
    markers = {"CATH reference": ".", "Protpardelle (diffusion)": "^", "VAE decoded": "s"}
    sizes = {"CATH reference": 15, "Protpardelle (diffusion)": 40, "VAE decoded": 40}
    alphas = {"CATH reference": 0.3, "Protpardelle (diffusion)": 0.8, "VAE decoded": 0.8}

    for label_name in ["CATH reference", "Protpardelle (diffusion)", "VAE decoded"]:
        mask = [l == label_name for l in labels]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=colors[label_name], marker=markers[label_name],
            s=sizes[label_name], alpha=alphas[label_name],
            label=label_name, edgecolors="none",
        )

    ax.set_xlabel(f"{method_name} 1", fontsize=11)
    ax.set_ylabel(f"{method_name} 2", fontsize=11)
    ax.set_title(f"VAE Latent Space: CATH vs Diffusion vs VAE Samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(outdir / "latent_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir / 'latent_umap.png'}")

    # Figure 2: Distance distributions comparison
    upper = np.triu_indices(CROP_SIZE, k=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ch_idx, ch_name, ax in zip([0, 1], ["CA-CA", "CB-CB"], axes):
        cath_dists = np.concatenate([s[ch_idx][upper] for s in cath_feats])
        diff_dists = np.concatenate([s[ch_idx][upper] for s in diff_feats])
        vae_dists = np.concatenate([s[ch_idx][upper] for s in vae_decoded])

        ax.hist(cath_dists, bins=80, density=True, alpha=0.5,
                label="CATH reference", color="#999999")
        ax.hist(diff_dists, bins=80, density=True, alpha=0.5,
                label="Protpardelle", color="#E07B54")
        ax.hist(vae_dists, bins=80, density=True, alpha=0.5,
                label="VAE decoded", color="#5B8DB8")
        ax.set_xlabel(f"Normalized {ch_name} distance", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{ch_name} Distance Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(outdir / "distance_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir / 'distance_distributions.png'}")

    # Figure 3: Contact density comparison
    cath_cd = [float(np.mean(s[2][upper] > 0.5)) for s in cath_feats]
    diff_cd = [float(np.mean(s[2][upper] > 0.5)) for s in diff_feats]
    vae_cd = [float(np.mean(s[2][upper] > 0.5)) for s in vae_decoded]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [cath_cd, diff_cd, vae_cd]
    labels_box = ["CATH reference", "Protpardelle\n(diffusion)", "VAE decoded"]
    colors_box = ["#999999", "#E07B54", "#5B8DB8"]

    bp = ax.boxplot(data, labels=labels_box, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Contact density", fontsize=11)
    ax.set_title("Contact Map Density by Method", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "contact_density_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir / 'contact_density_boxplot.png'}")

    # Figure 4: Feature channel means comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_CHANNELS)
    width = 0.25

    cath_means = [cath_feats[:, ch].mean() for ch in range(N_CHANNELS)]
    diff_means = [diff_feats[:, ch].mean() for ch in range(N_CHANNELS)]
    vae_means = [vae_decoded[:, ch].mean() for ch in range(N_CHANNELS)]

    ax.bar(x - width, cath_means, width, label="CATH reference", color="#999999", alpha=0.7)
    ax.bar(x, diff_means, width, label="Protpardelle", color="#E07B54", alpha=0.7)
    ax.bar(x + width, vae_means, width, label="VAE decoded", color="#5B8DB8", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, fontsize=8, rotation=15)
    ax.set_ylabel("Mean channel value", fontsize=11)
    ax.set_title("Feature Channel Means by Method", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "channel_means.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir / 'channel_means.png'}")

    # Figure 5: Sample distance maps side-by-side
    n_show = min(4, len(diff_feats), n_vae)
    fig, axes = plt.subplots(3, n_show, figsize=(3.5 * n_show, 9))
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_show):
        axes[0, i].imshow(cath_feats[i, 0], vmin=0, vmax=1, cmap="viridis")
        axes[0, i].set_title(f"CATH {i+1}", fontsize=9)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        axes[1, i].imshow(diff_feats[i, 0], vmin=0, vmax=1, cmap="viridis")
        axes[1, i].set_title(f"Protpardelle {i+1}", fontsize=9)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

        axes[2, i].imshow(vae_decoded[i, 0], vmin=0, vmax=1, cmap="viridis")
        axes[2, i].set_title(f"VAE {i+1}", fontsize=9)
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

    axes[0, 0].set_ylabel("CATH reference", fontsize=10)
    axes[1, 0].set_ylabel("Protpardelle", fontsize=10)
    axes[2, 0].set_ylabel("VAE decoded", fontsize=10)

    fig.suptitle("CA-CA Distance Maps: Reference vs Diffusion vs VAE",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "distance_map_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir / 'distance_map_comparison.png'}")

    # Summary metrics
    summary = {
        "CATH_reference": {
            "n_samples": len(cath_feats),
            "mean_contact_density": float(np.mean(cath_cd)),
            "std_contact_density": float(np.std(cath_cd)),
            "mean_ca_dist": float(np.mean([s[0][upper].mean() for s in cath_feats])),
        },
        "Protpardelle_diffusion": {
            "n_samples": len(diff_feats),
            "mean_contact_density": float(np.mean(diff_cd)),
            "std_contact_density": float(np.std(diff_cd)),
            "mean_ca_dist": float(np.mean([s[0][upper].mean() for s in diff_feats])),
        },
        "VAE_decoded": {
            "n_samples": n_vae,
            "mean_contact_density": float(np.mean(vae_cd)),
            "std_contact_density": float(np.std(vae_cd)),
            "mean_ca_dist": float(np.mean([s[0][upper].mean() for s in vae_decoded])),
        },
    }

    with open(outdir / "comparison_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'N':>5} {'Contact Density':>18} {'CA-CA Dist':>12}")
    print("-" * 70)
    for method, m in summary.items():
        print(f"{method:<25} {m['n_samples']:>5} "
              f"{m['mean_contact_density']:>8.4f} +/- {m['std_contact_density']:.4f} "
              f"{m['mean_ca_dist']:>12.4f}")

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
