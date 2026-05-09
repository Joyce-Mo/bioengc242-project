#!/usr/bin/env python
"""Evaluate trained VAE: generate samples, compute feature-level metrics, produce figures.

Generates N samples from a trained VAE checkpoint, computes quality metrics
on the 7-channel (C, L, L) feature maps, and creates presentation-ready
figures. Also loads sweep run checkpoints to compare configurations.

The 7 feature channels (from featurize_pdb.py / vae.py):
    ch 0: CA-CA distance map (normalized /D_MAX, clipped [0,1])
    ch 1: CB-CB distance map (normalized /D_MAX, clipped [0,1])
    ch 2: binary contact map (CA-CA < 8 angstrom)
    ch 3: hydrophobicity outer product (Kyte-Doolittle, [0,1])
    ch 4: charge outer product (-1/0/+1 shifted to [0,1])
    ch 5: polarity outer product (binary)
    ch 6: SASA outer product (normalized /SASA_MAX, clipped [0,1])

Metrics computed:
    - Distance map statistics (mean, std of CA-CA and CB-CB channels)
    - Contact density (fraction of nonzero entries in contact map channel)
    - Symmetry error (Frobenius norm of A - A^T, should be ~0)
    - Diagonal consistency (distance map diagonal should be ~0)
    - Channel correlation structure (cross-channel Pearson r)
    - Value range coverage per channel

Additionally, reconstructs 3D CA backbone coordinates from the generated
distance maps using classical multidimensional scaling (cMDS / distance
geometry). This allows saving PDB files from VAE samples and running the
same chi-angle KDE evaluation used for the diffusion model.

Reference for cMDS distance geometry:
    Torgerson, W.S. (1952). "Multidimensional scaling: I. Theory and method."
    Psychometrika, 17(4), 401-419.

Usage:
    python scripts/evaluation/evaluate_vae.py \
        --checkpoint /path/to/vae_best.pt \
        --training-samples /path/to/sample_*.npy \
        --outdir output/vae_eval \
        --num-samples 10

    # Sweep comparison:
    python scripts/evaluation/evaluate_vae.py \
        --sweep-dir /path/to/vae_sweep \
        --outdir output/vae_eval

    # With chi-angle evaluation against reference PDBs:
    python scripts/evaluation/evaluate_vae.py \
        --checkpoint /path/to/vae_best.pt \
        --outdir output/vae_eval \
        --num-samples 10 \
        --reference-pdb-dir /path/to/cath_pdbs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.distance import squareform, pdist

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel names for labeling figures
CHANNEL_NAMES = [
    "CA-CA distance",
    "CB-CB distance",
    "Contact map",
    "Hydrophobicity",
    "Charge",
    "Polarity",
    "SASA",
]

# Constants from vae.py
CROP_SIZE = 64
N_CHANNELS = 7
D_MAX = 22.0
SASA_MAX = 200.0


# VAE model definition (copied from vae/vae.py to avoid argparse side effects)
class VAE(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, z_dim=64,
                 dropout=0.0, use_batchnorm=False):
        super(VAE, self).__init__()
        self.dropout_p = dropout
        self.use_bn = use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)

        if use_batchnorm:
            self.bn_e1 = nn.BatchNorm2d(16)
            self.bn_e2 = nn.BatchNorm2d(32)
            self.bn_e3 = nn.BatchNorm2d(64)
            self.bn_e4 = nn.BatchNorm2d(128)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.dim_after_conv = 4
        self.hidden_dim = 128 * self.dim_after_conv * self.dim_after_conv

        self.fc21 = nn.Linear(self.hidden_dim, z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, self.hidden_dim)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1)

        if use_batchnorm:
            self.bn_d1 = nn.BatchNorm2d(64)
            self.bn_d2 = nn.BatchNorm2d(32)
            self.bn_d3 = nn.BatchNorm2d(16)

    def _enc_block(self, x, conv, bn=None):
        h = conv(x)
        if bn is not None:
            h = bn(h)
        return self.drop(F.relu(h))

    def _dec_block(self, x, deconv, bn=None):
        h = deconv(x)
        if bn is not None:
            h = bn(h)
        return self.drop(F.relu(h))

    def encode(self, x):
        h = self._enc_block(x, self.conv1, getattr(self, 'bn_e1', None))
        h = self._enc_block(h, self.conv2, getattr(self, 'bn_e2', None))
        h = self._enc_block(h, self.conv3, getattr(self, 'bn_e3', None))
        h = self._enc_block(h, self.conv4, getattr(self, 'bn_e4', None))
        h = h.view(h.size(0), -1)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.drop(F.relu(self.fc3(z)))
        h = h.view(h.size(0), 128, self.dim_after_conv, self.dim_after_conv)
        h = self._dec_block(h, self.deconv1, getattr(self, 'bn_d1', None))
        h = self._dec_block(h, self.deconv2, getattr(self, 'bn_d2', None))
        h = self._dec_block(h, self.deconv3, getattr(self, 'bn_d3', None))
        return torch.sigmoid(self.deconv4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Model loading
def load_vae(checkpoint_path, device="cpu"):
    """Load a trained VAE from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hparams", {})
    z_dim = ckpt.get("z_dim", hparams.get("z_dim", 64))
    use_bn = hparams.get("use_batchnorm", False)
    dropout = hparams.get("dropout", 0.0)

    model = VAE(
        in_channels=ckpt.get("in_channels", N_CHANNELS),
        z_dim=z_dim,
        dropout=dropout,
        use_batchnorm=use_bn,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, z_dim, hparams


def generate_samples(model, z_dim, n_samples=10, device="cpu", seed=42):
    """Generate n_samples by decoding random z vectors."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        samples = model.decode(z).cpu().numpy()
    return samples  # (n_samples, 7, 64, 64)


# Distance geometry: reconstruct 3D coordinates from distance maps
def classical_mds(dist_matrix, n_dims=3):
    """Classical multidimensional scaling (Torgerson, 1952).

    Reconstruct 3D coordinates from a pairwise distance matrix.
    Uses eigendecomposition of the doubly-centered squared distance matrix.

    Args:
        dist_matrix: (L, L) symmetric distance matrix in angstroms
        n_dims: number of output dimensions (3 for 3D coordinates)

    Returns:
        coords: (L, n_dims) array of reconstructed coordinates, or None if
                the distance matrix is degenerate
    """
    n = dist_matrix.shape[0]
    D_sq = dist_matrix ** 2

    # Double centering: B = -0.5 * J * D^2 * J, where J = I - (1/n) * 11^T
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # Eigendecomposition; take top n_dims positive eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # eigh returns ascending order, so take the last n_dims
    idx = np.argsort(eigenvalues)[::-1][:n_dims]
    evals = eigenvalues[idx]
    evecs = eigenvectors[:, idx]

    # Clamp negative eigenvalues to zero (noise from imperfect distance matrix)
    evals = np.maximum(evals, 0.0)

    if np.all(evals < 1e-10):
        return None

    coords = evecs * np.sqrt(evals)[np.newaxis, :]
    return coords.astype(np.float32)


def distance_map_to_ca_coords(ca_dist_normalized, d_max=D_MAX):
    """Convert a normalized CA-CA distance map back to 3D CA coordinates.

    Args:
        ca_dist_normalized: (L, L) array in [0, 1], from channel 0 of the VAE
        d_max: the D_MAX used for normalization (22.0 angstrom)

    Returns:
        coords: (L, 3) CA coordinates in angstroms, or None if degenerate
    """
    # Denormalize: multiply by D_MAX to get distances in angstroms
    dist_angstrom = ca_dist_normalized * d_max

    # Symmetrize (average with transpose) to clean up any asymmetry
    dist_angstrom = 0.5 * (dist_angstrom + dist_angstrom.T)

    # Zero out the diagonal
    np.fill_diagonal(dist_angstrom, 0.0)

    return classical_mds(dist_angstrom, n_dims=3)


def write_ca_trace_pdb(coords, pdb_path, chain_id="A"):
    """Write CA-only backbone trace as a PDB file.

    Assigns GLY as residue name since we only have CA positions.
    This is sufficient for distance-based metrics and visualization,
    though not for chi-angle analysis (which requires side chains).

    Args:
        coords: (L, 3) array of CA coordinates
        pdb_path: output path
        chain_id: chain identifier
    """
    with open(pdb_path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            atom_num = i + 1
            res_num = i + 1
            f.write(
                f"ATOM  {atom_num:5d}  CA  GLY {chain_id}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
            )
        f.write("END\n")


def samples_to_pdbs(samples, outdir, d_max=D_MAX):
    """Convert VAE samples to PDB files via distance geometry.

    Args:
        samples: (N, 7, 64, 64) array of generated feature maps
        outdir: directory to write PDB files
        d_max: D_MAX normalization constant

    Returns:
        list of (pdb_path, coords) tuples for successfully reconstructed samples
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []

    for i in range(samples.shape[0]):
        ca_dist = samples[i, 0]  # channel 0 = CA-CA distance map
        coords = distance_map_to_ca_coords(ca_dist, d_max=d_max)
        if coords is None:
            print(f"  Sample {i+1}: degenerate distance matrix, skipping PDB")
            continue

        pdb_path = outdir / f"vae_sample_{i+1:02d}.pdb"
        write_ca_trace_pdb(coords, pdb_path)
        results.append((str(pdb_path), coords))
        print(f"  Sample {i+1}: wrote {pdb_path}  ({coords.shape[0]} residues)")

    return results


def compute_reconstruction_rmsd(original_dist_normalized, reconstructed_coords, d_max=D_MAX):
    """Compute how well the reconstructed 3D coordinates reproduce the input distance map.

    This is a self-consistency check: if the distance geometry is working,
    the pairwise distances of the reconstructed coordinates should match
    the input distance map.

    Returns:
        rmsd: RMSD between input and reconstructed distance maps (angstroms)
    """
    original_dist = original_dist_normalized * d_max
    original_dist = 0.5 * (original_dist + original_dist.T)
    np.fill_diagonal(original_dist, 0.0)

    recon_dist = np.sqrt(((reconstructed_coords[:, None, :] -
                           reconstructed_coords[None, :, :]) ** 2).sum(axis=-1))

    # RMSD over upper triangle
    upper = np.triu_indices(original_dist.shape[0], k=1)
    diff = original_dist[upper] - recon_dist[upper]
    return float(np.sqrt(np.mean(diff ** 2)))


# Metrics
def compute_metrics(samples):
    """Compute per-sample and aggregate feature-level metrics.

    Args:
        samples: (N, 7, 64, 64) numpy array

    Returns:
        dict with per-sample and aggregate metrics
    """
    n = samples.shape[0]
    results = {"per_sample": [], "aggregate": {}}

    for i in range(n):
        s = samples[i]  # (7, 64, 64)
        ca_dist = s[0]
        cb_dist = s[1]
        contact = s[2]
        hydro = s[3]
        charge = s[4]
        polar = s[5]
        sasa = s[6]

        # Symmetry error: Frobenius norm of (A - A^T) / Frobenius norm of A
        # For protein feature maps, all channels should be symmetric
        sym_errors = []
        for ch in range(7):
            A = s[ch]
            frob_A = np.linalg.norm(A, "fro")
            if frob_A > 1e-8:
                sym_err = np.linalg.norm(A - A.T, "fro") / frob_A
            else:
                sym_err = 0.0
            sym_errors.append(float(sym_err))

        # Diagonal of distance maps should be ~0 (self-distance)
        ca_diag_mean = float(np.mean(np.diag(ca_dist)))
        cb_diag_mean = float(np.mean(np.diag(cb_dist)))

        # Contact density: fraction of upper triangle that is > 0.5
        upper = np.triu_indices(64, k=1)
        contact_density = float(np.mean(contact[upper] > 0.5))

        # Distance map statistics (upper triangle, excluding diagonal)
        ca_upper = ca_dist[upper]
        cb_upper = cb_dist[upper]

        # Per-channel value ranges
        ch_stats = {}
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            ch_data = s[ch_idx]
            ch_stats[ch_name] = {
                "mean": float(np.mean(ch_data)),
                "std": float(np.std(ch_data)),
                "min": float(np.min(ch_data)),
                "max": float(np.max(ch_data)),
            }

        results["per_sample"].append({
            "sample_idx": i,
            "symmetry_errors": sym_errors,
            "ca_diagonal_mean": ca_diag_mean,
            "cb_diagonal_mean": cb_diag_mean,
            "contact_density": contact_density,
            "ca_dist_mean": float(np.mean(ca_upper)),
            "ca_dist_std": float(np.std(ca_upper)),
            "cb_dist_mean": float(np.mean(cb_upper)),
            "cb_dist_std": float(np.std(cb_upper)),
            "channel_stats": ch_stats,
        })

    # Aggregate metrics across samples
    all_sym = np.array([s["symmetry_errors"] for s in results["per_sample"]])
    results["aggregate"] = {
        "n_samples": n,
        "mean_symmetry_error": {
            CHANNEL_NAMES[ch]: float(np.mean(all_sym[:, ch]))
            for ch in range(7)
        },
        "mean_contact_density": float(np.mean([s["contact_density"] for s in results["per_sample"]])),
        "std_contact_density": float(np.std([s["contact_density"] for s in results["per_sample"]])),
        "mean_ca_diagonal": float(np.mean([s["ca_diagonal_mean"] for s in results["per_sample"]])),
        "mean_ca_dist": float(np.mean([s["ca_dist_mean"] for s in results["per_sample"]])),
        "std_ca_dist": float(np.std([s["ca_dist_mean"] for s in results["per_sample"]])),
        "mean_cb_dist": float(np.mean([s["cb_dist_mean"] for s in results["per_sample"]])),
    }

    return results


def compute_metrics_on_training_samples(sample_paths):
    """Load saved training-time samples and compute metrics as reference."""
    all_samples = []
    for p in sample_paths:
        data = np.load(p)
        all_samples.append(data)
    samples = np.concatenate(all_samples, axis=0)
    return compute_metrics(samples), samples


# Figures
def fig_sample_grid(samples, outdir, prefix="production"):
    """Grid of CA-CA distance maps and contact maps for all samples.

    Creates two figures: one for distance maps (ch 0) and one for contacts (ch 2).
    """
    n = samples.shape[0]
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    for ch_idx, ch_label in [(0, "CA-CA distance"), (2, "Contact map")]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
        if nrows == 1:
            axes = axes[np.newaxis, :]
        for i in range(n):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            im = ax.imshow(samples[i, ch_idx], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"Sample {i+1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        # hide unused axes
        for i in range(n, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle(f"VAE Generated Samples: {ch_label}", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=axes, shrink=0.6, label="Normalized value")
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        safe_label = ch_label.replace(" ", "_").replace("-", "")
        fig.savefig(outdir / f"{prefix}_sample_grid_{safe_label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def fig_all_channels(sample, sample_idx, outdir, prefix="production"):
    """Visualize all 7 channels of a single sample."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    for ch in range(7):
        ax = axes_flat[ch]
        cmap = "viridis" if ch < 2 else ("Greys" if ch == 2 else "coolwarm")
        im = ax.imshow(sample[ch], vmin=0, vmax=1, cmap=cmap)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes_flat[7].set_visible(False)
    fig.suptitle(f"VAE Sample {sample_idx+1}: All Feature Channels", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / f"{prefix}_all_channels_sample{sample_idx+1}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_distance_distributions(samples, outdir, reference_samples=None, prefix="production"):
    """Compare distributions of CA-CA and CB-CB distances between generated and reference samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    upper = np.triu_indices(CROP_SIZE, k=1)

    for ch_idx, ch_name, ax in zip([0, 1], ["CA-CA", "CB-CB"], axes):
        # Generated samples
        gen_dists = np.concatenate([s[ch_idx][upper] for s in samples])
        ax.hist(gen_dists, bins=80, density=True, alpha=0.6, label="VAE generated", color="#5B8DB8")

        if reference_samples is not None:
            ref_dists = np.concatenate([s[ch_idx][upper] for s in reference_samples])
            ax.hist(ref_dists, bins=80, density=True, alpha=0.6, label="Training data (decoded)", color="#333333")

        ax.set_xlabel(f"Normalized {ch_name} distance", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{ch_name} Distance Distribution", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_distance_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_contact_density_bar(samples, outdir, reference_samples=None, prefix="production"):
    """Bar chart of contact density per sample, with reference mean if available."""
    upper = np.triu_indices(CROP_SIZE, k=1)
    densities = [float(np.mean(s[2][upper] > 0.5)) for s in samples]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(densities))
    ax.bar(x, densities, color="#5B8DB8", edgecolor="white", label="VAE samples")

    if reference_samples is not None:
        ref_densities = [float(np.mean(s[2][upper] > 0.5)) for s in reference_samples]
        ref_mean = np.mean(ref_densities)
        ax.axhline(ref_mean, color="#333333", linestyle="--", linewidth=2,
                    label=f"Training data mean ({ref_mean:.3f})")

    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_ylabel("Contact density", fontsize=11)
    ax.set_title("Contact Map Density per Sample", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in x])
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_contact_density.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_table(metrics, outdir, prefix="production"):
    """Create a figure with a summary metrics table for embedding in slides."""
    agg = metrics["aggregate"]

    rows = [
        ["Mean CA-CA distance (normalized)", f"{agg['mean_ca_dist']:.4f}"],
        ["Std CA-CA distance across samples", f"{agg['std_ca_dist']:.4f}"],
        ["Mean CB-CB distance (normalized)", f"{agg['mean_cb_dist']:.4f}"],
        ["Mean contact density", f"{agg['mean_contact_density']:.4f}"],
        ["Std contact density", f"{agg['std_contact_density']:.4f}"],
        ["Mean CA diagonal (should be ~0)", f"{agg['mean_ca_diagonal']:.4f}"],
    ]
    # Add symmetry errors
    for ch_name, val in agg["mean_symmetry_error"].items():
        rows.append([f"Symmetry error: {ch_name}", f"{val:.6f}"])

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.65, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D9E2F3")
    fig.suptitle("VAE Sample Quality Metrics", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / f"{prefix}_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_reconstruction_comparison(samples, pdb_results, outdir, prefix="production"):
    """Compare input distance maps with distance maps reconstructed from 3D coords.

    Shows that the distance geometry reconstruction is faithful to the VAE output.
    """
    n = min(len(pdb_results), 5)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        pdb_path, coords = pdb_results[i]
        input_dist = samples[i, 0]

        # Reconstructed distance map from 3D coords
        recon_dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
        recon_dist_norm = np.clip(recon_dist / D_MAX, 0, 1)

        # Difference
        diff = np.abs(input_dist - recon_dist_norm)

        axes[0, i].imshow(input_dist, vmin=0, vmax=1, cmap="viridis")
        axes[0, i].set_title(f"Input (sample {i+1})", fontsize=9)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        axes[1, i].imshow(recon_dist_norm, vmin=0, vmax=1, cmap="viridis")
        axes[1, i].set_title("Reconstructed", fontsize=9)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

        im = axes[2, i].imshow(diff, vmin=0, vmax=0.3, cmap="Reds")
        rmsd = compute_reconstruction_rmsd(input_dist, coords)
        axes[2, i].set_title(f"Diff (RMSD={rmsd:.1f} A)", fontsize=9)
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

    axes[0, 0].set_ylabel("VAE output", fontsize=10)
    axes[1, 0].set_ylabel("From 3D coords", fontsize=10)
    axes[2, 0].set_ylabel("|Difference|", fontsize=10)

    fig.suptitle("Distance Geometry Reconstruction Quality", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / f"{prefix}_reconstruction_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def featurize_reference_pdbs(pdb_dir, max_files=500):
    """Featurize reference PDB files into (C, L, L) feature stacks for comparison.

    Uses the same featurization as featurize_pdb.py but crops/pads to CROP_SIZE
    to match VAE output dimensions.
    """
    # Import featurize_pdb from the vae directory
    import importlib.util
    featurize_path = Path(__file__).resolve().parent.parent.parent / "vae" / "featurize_pdb.py"
    if not featurize_path.exists():
        print(f"  WARNING: featurize_pdb.py not found at {featurize_path}")
        return None

    spec = importlib.util.spec_from_file_location("featurize_pdb", featurize_path)
    fpdb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fpdb)

    pdb_files = sorted(Path(pdb_dir).rglob("*.pdb"))[:max_files]
    if not pdb_files:
        print(f"  No PDB files found in {pdb_dir}")
        return None

    print(f"  Featurizing {len(pdb_files)} reference PDBs...")
    features = []
    n_ok = 0
    for i, pdb_file in enumerate(pdb_files):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(pdb_files)}...")
        try:
            feats = fpdb.featurize_pdb(str(pdb_file))
        except Exception:
            continue

        # Crop/pad to CROP_SIZE (same logic as ProteinFeatureDataset)
        _, h, w = feats.shape
        if h >= CROP_SIZE:
            feats = feats[:, :CROP_SIZE, :CROP_SIZE]
        else:
            pad = CROP_SIZE - h
            feats = np.pad(feats, ((0, 0), (0, pad), (0, pad)), mode="constant")

        features.append(feats)
        n_ok += 1

    print(f"  Successfully featurized {n_ok}/{len(pdb_files)} PDBs")
    if not features:
        return None
    return np.stack(features, axis=0)


# Sweep comparison
def evaluate_sweep(sweep_dir, outdir, n_samples=10, device="cpu"):
    """Load each sweep checkpoint, generate samples, compute metrics, rank."""
    sweep_dir = Path(sweep_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_path = run_dir / "vae_best.pt"
        hparams_path = run_dir / "hparams.json"
        if not ckpt_path.exists():
            print(f"  SKIP {run_dir.name}: no vae_best.pt")
            continue

        print(f"  Evaluating {run_dir.name}...")
        hparams = json.loads(hparams_path.read_text()) if hparams_path.exists() else {}

        try:
            model, z_dim, _ = load_vae(ckpt_path, device=device)
            samples = generate_samples(model, z_dim, n_samples=n_samples, device=device)
            metrics = compute_metrics(samples)
        except Exception as e:
            print(f"    FAIL: {e}")
            continue

        agg = metrics["aggregate"]
        results.append({
            "run": run_dir.name,
            "z_dim": hparams.get("z_dim"),
            "lr": hparams.get("lr"),
            "weight_decay": hparams.get("weight_decay"),
            "kl_anneal_epochs": hparams.get("kl_anneal_epochs"),
            "use_batchnorm": hparams.get("use_batchnorm"),
            "mean_contact_density": agg["mean_contact_density"],
            "std_contact_density": agg["std_contact_density"],
            "mean_ca_dist": agg["mean_ca_dist"],
            "mean_ca_diagonal": agg["mean_ca_diagonal"],
            "mean_sym_error_ca": agg["mean_symmetry_error"]["CA-CA distance"],
        })

    if not results:
        print("No sweep runs found with checkpoints.")
        return None

    # Sort by contact density (higher = more structured) and low diagonal error
    # A good sample has: high contact density, low diagonal, low symmetry error
    for r in results:
        # Composite score: prefer low diagonal, low symmetry error, moderate contact density
        r["quality_score"] = (
            -abs(r["mean_ca_diagonal"])  # lower diagonal = better
            - r["mean_sym_error_ca"] * 10  # lower sym error = better
            + r["mean_contact_density"]  # some contact structure is good
        )
    results.sort(key=lambda r: r["quality_score"], reverse=True)

    # Save results
    with open(outdir / "sweep_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create sweep comparison figure
    fig_sweep_comparison(results, outdir)

    return results


def fig_sweep_comparison(results, outdir):
    """Create a figure comparing sweep configurations."""
    n = len(results)
    run_names = [r["run"] for r in results]
    # Shorten names for display
    short_names = []
    for name in run_names:
        # e.g. "z32_lr1e-3_dp0.2_wd0.0_kl0_bnfalse" -> "z32 lr=1e-3\nkl=0 bn=F"
        parts = name.split("_")
        z = parts[0] if parts else ""
        lr = parts[1].replace("lr", "lr=") if len(parts) > 1 else ""
        kl = ""
        bn = ""
        for p in parts:
            if p.startswith("kl"):
                kl = p.replace("kl", "kl=")
            if p.startswith("bn"):
                bn = "BN" if p == "bntrue" else ""
        label = f"{z} {lr}\n{kl} {bn}".strip()
        short_names.append(label)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Contact density
    ax = axes[0]
    colors = ["#5B8DB8" if i > 0 else "#E07B54" for i in range(n)]
    ax.barh(range(n), [r["mean_contact_density"] for r in results], color=colors, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Mean Contact Density", fontsize=10)
    ax.set_title("Contact Density", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # CA diagonal (should be ~0)
    ax = axes[1]
    ax.barh(range(n), [r["mean_ca_diagonal"] for r in results], color=colors, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Mean CA Diagonal Value", fontsize=10)
    ax.set_title("Diagonal Consistency (lower = better)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # Symmetry error
    ax = axes[2]
    ax.barh(range(n), [r["mean_sym_error_ca"] for r in results], color=colors, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Mean Symmetry Error (CA)", fontsize=10)
    ax.set_title("Symmetry Error (lower = better)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    fig.suptitle("Hyperparameter Sweep Comparison (ranked by quality score)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sweep comparison figure saved to {outdir / 'sweep_comparison.png'}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to vae_best.pt for the production run")
    parser.add_argument("--training-samples", type=str, nargs="*", default=None,
                        help="Paths to sample_*.npy from training (for reference distributions)")
    parser.add_argument("--sweep-dir", type=str, default=None,
                        help="Path to sweep directory for comparing configs")
    parser.add_argument("--reference-pdb-dir", type=str, default=None,
                        help="Directory of reference PDB files (e.g., AI-CATH) for "
                             "computing reference distance distributions")
    parser.add_argument("--reference-max-files", type=int, default=500,
                        help="Max reference PDBs to load (default: 500)")
    parser.add_argument("--outdir", type=str, default="output/vae_eval",
                        help="Output directory for figures and metrics")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Production model evaluation
    if args.checkpoint:
        print(f"Loading production VAE from {args.checkpoint}")
        model, z_dim, hparams = load_vae(args.checkpoint, device=args.device)
        print(f"  z_dim={z_dim}, epoch={hparams.get('epochs', '?')}")

        print(f"Generating {args.num_samples} samples...")
        samples = generate_samples(model, z_dim, n_samples=args.num_samples,
                                   device=args.device, seed=args.seed)
        np.save(outdir / "generated_samples.npy", samples)
        print(f"  Saved generated samples to {outdir / 'generated_samples.npy'}")

        print("Computing metrics...")
        metrics = compute_metrics(samples)

        # Load training-time reference samples if provided
        ref_samples = None
        if args.training_samples:
            print("Loading training-time reference samples...")
            ref_metrics, ref_samples = compute_metrics_on_training_samples(args.training_samples)
            metrics["reference"] = ref_metrics["aggregate"]

        with open(outdir / "production_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Creating figures...")
        fig_sample_grid(samples, outdir, prefix="production")
        fig_all_channels(samples[0], 0, outdir, prefix="production")
        if samples.shape[0] > 1:
            fig_all_channels(samples[1], 1, outdir, prefix="production")
        fig_distance_distributions(samples, outdir, reference_samples=ref_samples, prefix="production")
        fig_contact_density_bar(samples, outdir, reference_samples=ref_samples, prefix="production")
        fig_metrics_table(metrics, outdir, prefix="production")

        # Distance geometry: reconstruct 3D CA coordinates and save PDBs
        print("\nReconstructing 3D coordinates via classical MDS (distance geometry)...")
        pdb_dir = outdir / "generated_pdbs"
        pdb_results = samples_to_pdbs(samples, pdb_dir)

        # Compute reconstruction RMSD (self-consistency check)
        if pdb_results:
            recon_rmsds = []
            for (pdb_path, coords), sample in zip(pdb_results, samples):
                rmsd = compute_reconstruction_rmsd(sample[0], coords)
                recon_rmsds.append(rmsd)
            metrics["distance_geometry"] = {
                "n_reconstructed": len(pdb_results),
                "reconstruction_rmsds_angstrom": recon_rmsds,
                "mean_reconstruction_rmsd": float(np.mean(recon_rmsds)),
                "std_reconstruction_rmsd": float(np.std(recon_rmsds)),
            }
            print(f"  Reconstruction RMSD: {np.mean(recon_rmsds):.2f} +/- {np.std(recon_rmsds):.2f} angstrom")

            # Create figure comparing input vs reconstructed distance maps
            fig_reconstruction_comparison(samples, pdb_results, outdir, prefix="production")

        # Featurize reference PDBs if provided, for distribution comparison
        if args.reference_pdb_dir:
            print(f"\nFeaturizing reference PDBs from {args.reference_pdb_dir}...")
            ref_features = featurize_reference_pdbs(
                args.reference_pdb_dir, max_files=args.reference_max_files
            )
            if ref_features is not None:
                ref_metrics_struct = compute_metrics(ref_features)
                metrics["reference_pdb"] = ref_metrics_struct["aggregate"]
                fig_distance_distributions(
                    samples, outdir,
                    reference_samples=ref_features,
                    prefix="production_vs_cath",
                )
                fig_contact_density_bar(
                    samples, outdir,
                    reference_samples=ref_features,
                    prefix="production_vs_cath",
                )

        # Re-save metrics with distance geometry results
        with open(outdir / "production_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\nProduction model metrics summary:")
        agg = metrics["aggregate"]
        print(f"  Mean contact density: {agg['mean_contact_density']:.4f} +/- {agg['std_contact_density']:.4f}")
        print(f"  Mean CA-CA distance:  {agg['mean_ca_dist']:.4f} +/- {agg['std_ca_dist']:.4f}")
        print(f"  Mean CA diagonal:     {agg['mean_ca_diagonal']:.4f}")
        print(f"  Mean symmetry errors:")
        for ch, val in agg["mean_symmetry_error"].items():
            print(f"    {ch}: {val:.6f}")

    # Sweep comparison
    if args.sweep_dir:
        print(f"\nEvaluating sweep runs from {args.sweep_dir}...")
        sweep_results = evaluate_sweep(
            args.sweep_dir, outdir,
            n_samples=args.num_samples, device=args.device,
        )
        if sweep_results:
            print("\nSweep ranking (best first):")
            for i, r in enumerate(sweep_results):
                print(f"  {i+1}. {r['run']}  contact={r['mean_contact_density']:.4f}  "
                      f"diag={r['mean_ca_diagonal']:.4f}  sym={r['mean_sym_error_ca']:.6f}")

    print(f"\nAll outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
