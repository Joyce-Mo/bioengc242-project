"""One-step diffusion structure module conditioned on VAE feature maps.

Couples the VAE decoder (which outputs 7-channel (C, L, L) protein feature
maps) with a lightweight denoising network that predicts 3D CA backbone
coordinates. The denoising network is trained with either:

    1. Flow matching loss (Lipman et al., 2023; used by La Proteina / Proteina)
       - Linear interpolation: x_t = (1-t)*noise + t*clean
       - Model predicts clean coords from x_t
       - Loss weight: 1/(1-t)^2

    2. Score-based / EDM loss (Karras et al., 2022; used by Protpardelle)
       - Gaussian perturbation: x_noisy = clean + sigma*noise
       - Model predicts clean coords from x_noisy
       - Loss weight: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

Both follow the implementations in protpardelle-1c/src/protpardelle/train.py.

Architecture:
    The structure module extracts per-residue features from the (7, L, L)
    feature map via row-wise pooling and diagonal extraction, concatenates
    these with noisy 3D coordinates, and predicts denoised coordinates
    through a residual MLP.

Usage:
    # Train:
    python vae/structure_module.py \
        --pdb-dir /path/to/cath_pdbs \
        --vae-checkpoint /path/to/vae_best.pt \
        --loss-type flow_matching \
        --outdir output/structure_module

    # Sample (after training):
    python vae/structure_module.py \
        --sample \
        --vae-checkpoint /path/to/vae_best.pt \
        --struct-checkpoint output/structure_module/struct_best.pt \
        --num-samples 10 \
        --outdir output/structure_module/samples
"""

from __future__ import print_function
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Constants (must match vae.py / featurize_pdb.py)
CROP_SIZE = 64
N_CHANNELS = 7
D_MAX = 22.0
SASA_MAX = 200.0

CHANNEL_NAMES = [
    "CA-CA dist", "CB-CB dist", "Contact",
    "Hydrophobicity", "Charge", "Polarity", "SASA",
]


# VAE model (copied from vae.py to avoid argparse side effects at import)
class VAE(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, z_dim=64,
                 dropout=0.0, use_batchnorm=False):
        super().__init__()
        self.dropout_p = dropout
        self.use_bn = use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, 16, 4, padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 4, padding=1, stride=2)

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
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1)

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


def load_vae(checkpoint_path, device="cpu"):
    """Load a trained VAE from checkpoint."""
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
    model.to(device)
    model.eval()
    return model, z_dim


# Feature extraction: (7, L, L) feature map -> (L, D) per-residue features
class FeatureMapToResidueFeatures(nn.Module):
    """Extract per-residue feature vectors from a (C, L, L) pair feature map.

    Three sources of per-residue information:
        1. Diagonal: self-interaction features (C values per residue)
        2. Row-wise mean pooling: average context from all partners (C values)
        3. Row-wise max pooling: strongest interaction signal (C values)

    Output: (L, 3*C) = (L, 21) per-residue feature vector, projected to d_model.
    """

    def __init__(self, n_channels=N_CHANNELS, d_model=128):
        super().__init__()
        self.proj = nn.Linear(3 * n_channels, d_model)

    def forward(self, feat_map):
        """
        Args:
            feat_map: (B, C, L, L) feature map from VAE decoder
        Returns:
            (B, L, d_model) per-residue features
        """
        # Diagonal: (B, C, L) -> (B, L, C)
        diag = torch.diagonal(feat_map, dim1=-2, dim2=-1)  # (B, C, L)
        diag = diag.permute(0, 2, 1)  # (B, L, C)

        # Row-wise pooling: (B, C, L, L) -> (B, C, L) -> (B, L, C)
        row_mean = feat_map.mean(dim=-1).permute(0, 2, 1)
        row_max = feat_map.max(dim=-1).values.permute(0, 2, 1)

        # Concatenate: (B, L, 3*C)
        per_res = torch.cat([diag, row_mean, row_max], dim=-1)
        return self.proj(per_res)  # (B, L, d_model)


# Structure module: predict CA coordinates from per-residue features + noise
class StructureModule(nn.Module):
    """Denoising network that predicts clean CA coordinates.

    Takes per-residue features (from the VAE feature map) and noisy
    coordinates, and predicts the denoised (clean) CA coordinates.

    Architecture: residual MLP with sinusoidal time/noise embedding,
    following the simplicity of the original VAE project style.
    """

    def __init__(self, d_model=128, n_layers=6, d_time=64):
        super().__init__()
        self.feat_extractor = FeatureMapToResidueFeatures(
            n_channels=N_CHANNELS, d_model=d_model,
        )

        # Sinusoidal time/noise level embedding
        self.d_time = d_time
        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Coordinate embedding: (x, y, z) -> d_model
        self.coord_embed = nn.Linear(3, d_model)

        # Residual MLP blocks
        # Input: feat + coord + time = d_model (summed)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            ))

        # Output projection: d_model -> 3 (x, y, z)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 3)

    def sinusoidal_embedding(self, t):
        """Sinusoidal positional embedding for scalar t values.

        Args:
            t: (B,) tensor of timesteps or noise levels

        Returns:
            (B, d_time) embedding
        """
        half = self.d_time // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, feat_map, noisy_coords, t):
        """
        Args:
            feat_map: (B, C, L, L) VAE-decoded feature map
            noisy_coords: (B, L, 3) noisy CA coordinates
            t: (B,) timestep (flow matching) or noise level (EDM)

        Returns:
            (B, L, 3) predicted clean CA coordinates
        """
        # Per-residue features from the 2D feature map
        h_feat = self.feat_extractor(feat_map)  # (B, L, d_model)

        # Coordinate embedding
        h_coord = self.coord_embed(noisy_coords)  # (B, L, d_model)

        # Time embedding (broadcast to all residues)
        h_time = self.time_proj(self.sinusoidal_embedding(t))  # (B, d_model)
        h_time = h_time.unsqueeze(1)  # (B, 1, d_model)

        # Sum conditioning signals
        h = h_feat + h_coord + h_time

        # Residual MLP blocks
        for block in self.blocks:
            h = h + block(h)

        # Predict clean coordinates
        return self.out_proj(self.out_norm(h))


# Loss functions (adapted from protpardelle-1c/src/protpardelle/train.py)
def flow_matching_loss(model, feat_map, clean_coords, tol=1e-5):
    """Flow matching loss (Lipman et al., 2023).

    Linear interpolation between noise and clean data:
        x_t = (1 - t) * eps + t * x_clean
    where t ~ U(0, 1) and eps ~ N(0, I).

    The model predicts x_clean from x_t. Loss is weighted MSE with
    weight 1/(1-t)^2, which emphasizes predictions near clean data
    (high t) where accuracy matters most for generation quality.

    References:
        - Lipman et al. "Flow Matching for Generative Modeling" (ICLR 2023)
        - protpardelle-1c train.py lines 757-916
    """
    B, L, _ = clean_coords.shape
    device = clean_coords.device

    # Sample t ~ U(tol, 1-tol)
    t = torch.rand(B, device=device).clamp(min=tol, max=1.0 - tol)

    # Pure noise (x_0 in flow matching notation)
    eps = torch.randn_like(clean_coords)

    # Linear interpolation: x_t = (1-t)*eps + t*clean
    t_expand = t[:, None, None]  # (B, 1, 1)
    x_t = (1.0 - t_expand) * eps + t_expand * clean_coords

    # Model predicts clean coordinates from noisy x_t
    pred_clean = model(feat_map, x_t, t)

    # Flow matching weight: 1/(1-t)^2
    weight = 1.0 / ((1.0 - t).square() + tol)  # (B,)
    weight = weight[:, None, None]  # (B, 1, 1)

    # Weighted MSE
    mse = (pred_clean - clean_coords).square()  # (B, L, 3)
    loss = (weight * mse).mean()

    return loss


def score_based_loss(model, feat_map, clean_coords, sigma_data=10.0, tol=1e-5):
    """EDM / score-based denoising loss (Karras et al., 2022).

    Gaussian perturbation with lognormal noise schedule:
        sigma ~ lognormal(P_mean, P_std)
        x_noisy = x_clean + sigma * eps

    The model predicts x_clean from x_noisy. Loss is weighted MSE with
    the Karras et al. preconditioning weight:
        w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

    This weight balances the loss across noise levels: at high noise
    (large sigma), the target is hard but the weight is small; at low
    noise, the prediction is easier but weighted more heavily.

    References:
        - Karras et al. "Elucidating the Design Space of Diffusion-Based
          Generative Models" (NeurIPS 2022)
        - protpardelle-1c train.py lines 918-1126
    """
    B, L, _ = clean_coords.shape
    device = clean_coords.device

    # Lognormal noise schedule (same as protpardelle training)
    # P_mean = -0.5, P_std = 1.5 are the defaults from Karras et al.
    t = torch.rand(B, device=device).clamp(min=tol, max=1.0 - tol)
    normal_sample = torch.erfinv(2.0 * t - 1.0) * math.sqrt(2.0)
    sigma = sigma_data * torch.exp(-0.5 + 1.5 * normal_sample)
    sigma = sigma.clamp(min=tol)

    # Gaussian perturbation
    eps = torch.randn_like(clean_coords)
    sigma_expand = sigma[:, None, None]  # (B, 1, 1)
    x_noisy = clean_coords + sigma_expand * eps

    # Model predicts clean coordinates
    pred_clean = model(feat_map, x_noisy, sigma)

    # Karras et al. preconditioning weight
    sigma_data_t = torch.tensor(sigma_data, device=device, dtype=sigma.dtype)
    denom = (sigma * sigma_data_t).square().clamp(min=tol)
    weight = (sigma.square() + sigma_data_t.square()) / denom  # (B,)
    weight = weight[:, None, None]  # (B, 1, 1)

    # Weighted MSE
    mse = (pred_clean - clean_coords).square()
    loss = (weight * mse).mean()

    return loss


# Dataset: paired (feature_map, CA_coords) from PDB files
class PairedFeatureCoordDataset(torch.utils.data.Dataset):
    """Dataset that provides paired (feature_map, CA_coords) for training.

    Each item is:
        - feature_map: (7, L, L) from featurize_pdb.py, cropped/padded to CROP_SIZE
        - ca_coords: (L, 3) CA coordinates in angstroms, cropped/padded to match

    The CA coordinates are centered (mean-subtracted) for translation invariance,
    following protpardelle's convention.
    """

    def __init__(self, pdb_paths, crop_size=CROP_SIZE):
        self.pdb_paths = list(pdb_paths)
        self.crop_size = crop_size
        # Lazy import to avoid circular dependency
        self._featurizer = None

    def _get_featurizer(self):
        if self._featurizer is None:
            import importlib.util
            fp = Path(__file__).resolve().parent / "featurize_pdb.py"
            spec = importlib.util.spec_from_file_location("featurize_pdb", fp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._featurizer = mod
        return self._featurizer

    def __len__(self):
        return len(self.pdb_paths)

    def _extract_ca_coords(self, pdb_path):
        """Extract CA coordinates from a PDB file."""
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = parser.get_structure("s", str(pdb_path))
        model = next(structure.get_models())
        chain = next(model.get_chains())
        coords = []
        for residue in chain:
            if residue.id[0].strip() != "":
                continue
            if "CA" not in residue:
                continue
            coords.append(np.asarray(residue["CA"].get_coord(), dtype=np.float32))
        return np.stack(coords) if coords else None

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        fpdb = self._get_featurizer()

        # Get feature map
        feats = fpdb.featurize_pdb(str(pdb_path)).astype(np.float32)

        # Get CA coordinates
        ca_coords = self._extract_ca_coords(pdb_path)
        if ca_coords is None:
            raise ValueError(f"No CA atoms in {pdb_path}")

        L = ca_coords.shape[0]
        assert feats.shape[1] == L, f"Feature L={feats.shape[1]} != coord L={L}"

        # Crop or pad to CROP_SIZE (same logic as vae.py)
        if L >= self.crop_size:
            # Random crop
            i = np.random.randint(0, L - self.crop_size + 1)
            feats = feats[:, i:i + self.crop_size, i:i + self.crop_size]
            ca_coords = ca_coords[i:i + self.crop_size]
        else:
            # Pad features with zeros
            pad = self.crop_size - L
            feats = np.pad(feats, ((0, 0), (0, pad), (0, pad)), mode="constant")
            # Pad coordinates with the centroid (will be zeroed after centering)
            ca_coords = np.pad(ca_coords, ((0, pad), (0, 0)), mode="constant")

        # Center CA coordinates on their centroid (translation invariance)
        # Only center on real (non-padded) residues
        real_L = min(L, self.crop_size)
        centroid = ca_coords[:real_L].mean(axis=0, keepdims=True)
        ca_coords[:real_L] -= centroid

        # Create mask: 1 for real residues, 0 for padding
        mask = np.zeros(self.crop_size, dtype=np.float32)
        mask[:real_L] = 1.0

        return (
            torch.from_numpy(feats),
            torch.from_numpy(ca_coords),
            torch.from_numpy(mask),
        )


# Sampling
@torch.no_grad()
def sample_flow_matching(struct_model, feat_map, num_steps=100):
    """Generate CA coordinates via flow matching ODE integration.

    Euler integration from t=0 (pure noise) to t=1 (clean data):
        x_0 ~ N(0, I)
        dx/dt = (x_hat - x_t) / (1 - t)  where x_hat = model(feat_map, x_t, t)
        x_{t+dt} = x_t + dt * dx/dt

    Args:
        struct_model: trained StructureModule
        feat_map: (B, 7, L, L) VAE-decoded feature map
        num_steps: number of Euler steps

    Returns:
        (B, L, 3) generated CA coordinates
    """
    B = feat_map.shape[0]
    L = feat_map.shape[2]
    device = feat_map.device

    # Start from pure noise
    x = torch.randn(B, L, 3, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((B,), t_val, device=device).clamp(min=1e-5, max=1.0 - 1e-5)

        # Model predicts clean data
        x_hat = struct_model(feat_map, x, t)

        # Flow matching velocity: (x_hat - x) / (1 - t)
        velocity = (x_hat - x) / (1.0 - t_val + 1e-5)

        x = x + dt * velocity

    return x


@torch.no_grad()
def sample_edm(struct_model, feat_map, num_steps=100, sigma_data=10.0,
               s_min=0.001, s_max=80.0, rho=7.0):
    """Generate CA coordinates via EDM sampling (Karras et al. 2022).

    Deterministic sampler using the Karras et al. noise schedule:
        sigma_i = (s_max^(1/rho) + i/(N-1) * (s_min^(1/rho) - s_max^(1/rho)))^rho

    At each step, the model denoises and we step to the next noise level.

    Args:
        struct_model: trained StructureModule
        feat_map: (B, 7, L, L) VAE-decoded feature map
        num_steps: number of denoising steps

    Returns:
        (B, L, 3) generated CA coordinates
    """
    B = feat_map.shape[0]
    L = feat_map.shape[2]
    device = feat_map.device

    # Karras noise schedule: decreasing sigma levels
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float32)
    sigmas = (
        s_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])  # append sigma=0

    # Start from noise at sigma_max
    x = torch.randn(B, L, 3, device=device) * sigmas[0]

    for i in range(num_steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_batch = torch.full((B,), sigma_cur.item(), device=device)
        x_hat = struct_model(feat_map, x, sigma_batch)

        # Step toward denoised prediction
        # d = (x - x_hat) / sigma_cur  (score direction)
        # x_next = x_hat + sigma_next * (x - x_hat) / sigma_cur
        if sigma_next > 0:
            x = x_hat + sigma_next * (x - x_hat) / sigma_cur
        else:
            x = x_hat

    return x


# PDB writing (all-atom not available, CA trace only)
def write_ca_pdb(coords, path, chain_id="A"):
    """Write CA-only backbone trace as PDB. coords: (L, 3) numpy array."""
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(
                f"ATOM  {i+1:5d}  CA  GLY {chain_id}{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
            )
        f.write("END\n")


# Training loop
def train(args):
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load frozen VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae, z_dim = load_vae(args.vae_checkpoint, device=device)
    for p in vae.parameters():
        p.requires_grad_(False)

    # Build dataset
    pdb_paths = sorted(str(p) for p in Path(args.pdb_dir).rglob("*.pdb"))
    if not pdb_paths:
        raise SystemExit(f"No PDB files found under {args.pdb_dir}")
    if args.max_pdbs:
        pdb_paths = pdb_paths[:args.max_pdbs]
    print(f"Found {len(pdb_paths)} PDB files")

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_paths, val_paths = train_test_split(
        pdb_paths, test_size=0.15, random_state=args.seed,
    )

    # Custom collate to skip failed PDBs
    def safe_collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataset = PairedFeatureCoordDataset(train_paths)
    val_dataset = PairedFeatureCoordDataset(val_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=safe_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=safe_collate,
    )

    # Build structure module
    struct_model = StructureModule(
        d_model=args.d_model, n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in struct_model.parameters())
    print(f"Structure module: {n_params:,} parameters")
    print(f"Loss type: {args.loss_type}")

    optimizer = optim.AdamW(
        struct_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Select loss function
    if args.loss_type == "flow_matching":
        loss_fn = lambda model, fm, coords: flow_matching_loss(model, fm, coords)
    elif args.loss_type == "score_based":
        loss_fn = lambda model, fm, coords: score_based_loss(
            model, fm, coords, sigma_data=args.sigma_data,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Save hyperparameters
    hparams = vars(args).copy()
    hparams["n_params"] = n_params
    (outdir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    history = {"train": [], "val": [], "lr": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        struct_model.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            if batch is None:
                continue
            feats, coords, mask = batch
            feats = feats.to(device)
            coords = coords.to(device)
            mask = mask.to(device)

            # Get VAE reconstruction of the feature map (frozen VAE)
            with torch.no_grad():
                recon_feats, _, _ = vae(feats)

            optimizer.zero_grad()
            loss = loss_fn(struct_model, recon_feats, coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(struct_model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * feats.shape[0]
            n_train += feats.shape[0]

        scheduler.step()
        train_avg = train_loss_sum / max(n_train, 1)

        # Validate
        struct_model.eval()
        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                feats, coords, mask = batch
                feats = feats.to(device)
                coords = coords.to(device)

                recon_feats, _, _ = vae(feats)
                loss = loss_fn(struct_model, recon_feats, coords)
                val_loss_sum += loss.item() * feats.shape[0]
                n_val += feats.shape[0]

        val_avg = val_loss_sum / max(n_val, 1)

        history["train"].append(train_avg)
        history["val"].append(val_avg)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_avg:.4f}  val={val_avg:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                "state_dict": struct_model.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "loss_type": args.loss_type,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "sigma_data": args.sigma_data,
                "hparams": hparams,
            }, outdir / "struct_best.pt")

    # Save history
    (outdir / "history.json").write_text(json.dumps(history, indent=2))

    # Plot training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="Train", alpha=0.8)
    ax.plot(history["val"], label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Structure Module Training ({args.loss_type})")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outdir / "training_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Outputs in {outdir}")


# Sampling mode
def sample(args):
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae, z_dim = load_vae(args.vae_checkpoint, device=device)

    # Load structure module
    print(f"Loading structure module from {args.struct_checkpoint}")
    ckpt = torch.load(args.struct_checkpoint, map_location=device, weights_only=False)
    struct_model = StructureModule(
        d_model=ckpt.get("d_model", 128),
        n_layers=ckpt.get("n_layers", 6),
    ).to(device)
    struct_model.load_state_dict(ckpt["state_dict"])
    struct_model.eval()

    loss_type = ckpt.get("loss_type", args.loss_type)
    sigma_data = ckpt.get("sigma_data", args.sigma_data)
    print(f"  Loss type: {loss_type}, sigma_data: {sigma_data}")

    # Generate feature maps from VAE
    print(f"Generating {args.num_samples} samples...")
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.num_samples, z_dim, device=device)
        feat_maps = vae.decode(z)

    # Generate coordinates
    if loss_type == "flow_matching":
        coords = sample_flow_matching(
            struct_model, feat_maps, num_steps=args.num_steps,
        )
    else:
        coords = sample_edm(
            struct_model, feat_maps, num_steps=args.num_steps,
            sigma_data=sigma_data,
        )

    coords_np = coords.cpu().numpy()

    # Save PDB files
    pdb_dir = outdir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for i in range(args.num_samples):
        pdb_path = pdb_dir / f"vae_struct_sample_{i+1:02d}.pdb"
        write_ca_pdb(coords_np[i], pdb_path)
        print(f"  Wrote {pdb_path}")

    # Save raw coordinates
    np.save(outdir / "generated_coords.npy", coords_np)

    # Save feature maps
    np.save(outdir / "generated_feat_maps.npy", feat_maps.cpu().numpy())

    # Visualize: distance maps from generated coordinates vs VAE feature maps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(args.num_samples, 5)
    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_show):
        # VAE feature map (channel 0 = CA-CA distance)
        axes[0, i].imshow(feat_maps[i, 0].cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
        axes[0, i].set_title(f"VAE dist map {i+1}", fontsize=9)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        # Distance map from generated coordinates
        c = coords_np[i]
        recon_dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=-1))
        recon_dist_norm = np.clip(recon_dist / D_MAX, 0, 1)
        axes[1, i].imshow(recon_dist_norm, vmin=0, vmax=1, cmap="viridis")
        axes[1, i].set_title(f"Generated 3D {i+1}", fontsize=9)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

    axes[0, 0].set_ylabel("VAE decoder", fontsize=10)
    axes[1, 0].set_ylabel("Structure module", fontsize=10)
    fig.suptitle(f"VAE + Structure Module Samples ({loss_type})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "sample_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs saved to {outdir}")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="One-step diffusion structure module conditioned on VAE feature maps"
    )
    sub = parser.add_subparsers(dest="mode")

    # Train subcommand
    train_p = sub.add_parser("train", help="Train the structure module")
    train_p.add_argument("--pdb-dir", type=str, required=True,
                         help="Directory of PDB files for training")
    train_p.add_argument("--vae-checkpoint", type=str, required=True,
                         help="Path to trained VAE checkpoint (vae_best.pt)")
    train_p.add_argument("--loss-type", choices=["flow_matching", "score_based"],
                         default="flow_matching",
                         help="Loss function: flow_matching or score_based (default: flow_matching)")
    train_p.add_argument("--sigma-data", type=float, default=10.0,
                         help="Data scale for EDM loss (default: 10.0, from protpardelle)")
    train_p.add_argument("--outdir", type=str, default="output/structure_module")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--d-model", type=int, default=128,
                         help="Hidden dimension (default: 128)")
    train_p.add_argument("--n-layers", type=int, default=6,
                         help="Number of residual MLP layers (default: 6)")
    train_p.add_argument("--max-pdbs", type=int, default=None,
                         help="Limit number of training PDBs (for quick testing)")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", type=str, default="cpu")

    # Sample subcommand
    sample_p = sub.add_parser("sample", help="Generate samples from trained model")
    sample_p.add_argument("--vae-checkpoint", type=str, required=True)
    sample_p.add_argument("--struct-checkpoint", type=str, required=True)
    sample_p.add_argument("--loss-type", choices=["flow_matching", "score_based"],
                          default="flow_matching")
    sample_p.add_argument("--sigma-data", type=float, default=10.0)
    sample_p.add_argument("--num-samples", type=int, default=10)
    sample_p.add_argument("--num-steps", type=int, default=100,
                          help="Number of ODE/denoising steps (default: 100)")
    sample_p.add_argument("--outdir", type=str, default="output/structure_module/samples")
    sample_p.add_argument("--seed", type=int, default=42)
    sample_p.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
