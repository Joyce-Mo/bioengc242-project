"""Protein VAE for multi-conformer structure encoding/generation.

Adapted from BIOENGC242 Tutorial 10's image VAE for the protein-VAE branch
of the project. The model encodes per-residue spatial features (amino-acid polarity / hydrophobicity /
charge, solvent acccessible surface area (SASA)) and pairwise distance maps into some sort of
low dimensional latent "z", and decodes z back to a 2D feature.

Layout of this script for purposes of running on an HPC follows the PyTorch VAE example
(https://github.com/pytorch/examples/blob/main/vae/main.py)
"""

from __future__ import print_function
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


# # Protein feature representation
#
# Each protein conformer is represented as a (C, L, L) tensor with protein-specific channels
#
#   ch 0  : Calpha-Calpha distance map (angstrom, normalized to [0, 1] by /D_MAX)
#   ch 1  : Cbeta-Cbeta distance map (angstrom, normalized to [0, 1] by /D_MAX)
#   ch 2  : binary contact map (Calpha-Calpha < 8 angstrom)
#   ch 3  : hydrophobicity outer product   (Kyte–Doolittle, scaled to [0, 1])
#   ch 4  : charge outer product           (-1 / 0 / +1, shifted to [0, 1])
#   ch 5  : polarity outer product         (binary, polar vs nonpolar)
#   ch 6  : per-residue SASA outer product (normalized to [0, 1] by SASA_MAX)
#
# Inputs are cropped / padded to a fixed window of L = CROP_SIZE residues so
# the conv stack has a deterministic shape, similar to tutorial 10 which uses
# 32x32 MNIST. CATH domains in the AI-CATH subset that I am using as the trainind data
# typically span 60–200 residues, so a 64-residue window covers most short domains and gives a
# random crop for longer ones.


# PARAMS THAT MGITH BE TUNED LATER!! durign the next checkpoint when i will 
# do a grid search. These are somewhat random. Although I will probably keep number of channels the 
# same. CROP_SIZE, D_MAX, SASA_MAX will be updated based on distributions of AI_cath 
# dataset after I generate the rest of the MCSCE side chain conformers.
CROP_SIZE = 64        # L - residues per window
N_CHANNELS = 7        # C - feature channels described above
D_MAX = 22.0          # angstrom - distance map clip / normalization
SASA_MAX = 200.0      # angstrom^2 - per-residue SASA normalization


# Parsers for scripts. 
parser = argparse.ArgumentParser(description='Protein VAE')
parser.add_argument('--feature-dir', type=str, default=None,
                    help='directory of (C, L, L) .npy feature stacks; '
                         'omit to run the random-tensor smoke test')
parser.add_argument('--outdir', type=str, default='output/vae',
                    help='where to write checkpoints / metrics / samples')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--z-dim', type=int, default=64, metavar='Z',
                    help='dimensionality of the latent space (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Adam learning rate (default: 1e-3)')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

use_accel = not args.no_accel and torch.accelerator.is_available()

torch.manual_seed(args.seed)


if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# # Protein feature dataset
#
# A small datset stub that takes a list of (C, L, L) feature tensors
# pre-computed from PDBs and yields random L*L crops. The actual feature
# extraction (Calpha/Cbeta distance maps, hydrophobicity / charge / polarity
# outer products, SASA via DSSP) lives in `scripts/featurize_pdb.py` (TODO)
# so that this file stays focused on the model. 
# 
# The 2nd return value is a dummy label so the (data, target) destructuring in train/test matches the
# torchvision convention used by the pytorch example.

class ProteinFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_paths, crop_size=CROP_SIZE):
        # feature_paths : list of .npy files, each shape (C, L_i, L_i),
        #  scaled to [0, 1]
        self.feature_paths = list(feature_paths)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        # load (C, L, L) feature stack for one conformer
        feats = np.load(self.feature_paths[idx]).astype(np.float32)
        _, h, w = feats.shape
        assert h == w, f"expected square feature map, got {h}x{w}"
        # random crop (or pad) to crop_size
        if h >= self.crop_size:
            i = np.random.randint(0, h - self.crop_size + 1)
            feats = feats[:, i:i + self.crop_size, i:i + self.crop_size]
        else:
            pad = self.crop_size - h
            feats = np.pad(feats, ((0, 0), (0, pad), (0, pad)), mode="constant")
        return torch.from_numpy(feats), 0  # dummy label for now... matches (data, target)


# Build the train / val / test loaders. The training/val/testing split might be updated later 
# too. For now, it's 70/15/15. 
kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
if args.feature_dir is not None:
    feature_paths = sorted(str(p) for p in Path(args.feature_dir).rglob("*.npy"))
    if not feature_paths:
        raise SystemExit(f"no .npy feature files found under {args.feature_dir}")
    print(f"found {len(feature_paths)} feature files under {args.feature_dir}")

    train_paths, holdout_paths = train_test_split(
        feature_paths, test_size=0.30, random_state=args.seed,
    )
    val_paths, test_paths = train_test_split(
        holdout_paths, test_size=0.50, random_state=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        ProteinFeatureDataset(train_paths),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        ProteinFeatureDataset(val_paths),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        ProteinFeatureDataset(test_paths),
        batch_size=args.batch_size, shuffle=False, **kwargs)
else:
    # No data given == smoke-test mode (see __main__).
    train_loader = val_loader = test_loader = None


## Variational Autoencoder (VAE)
# VAEs provide an unsupervised way to compress data.
# The decoder of a VAE can be used as a generative model.

class VAE(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, z_dim=args.z_dim):
        super(VAE, self).__init__()

        # encoder: 64 to 32 to 16 to 8 to 4
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, padding=1, stride=2)  # conv1, in to 16    (64 to 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)           # conv2, 16 to 32    (32 to 16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)           # conv3, 32 to 64    (16 to 8)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)          # conv4, 64 to 128   (8 to 4)

        # manually calculate the dimension after all convolutions
        self.dim_after_conv = 4
        self.hidden_dim = 128 * self.dim_after_conv * self.dim_after_conv

        # readout: parameterize log(sigma^2) instead of sigma directly so the
        # network output can take any real value (sigma^2 = exp(logvar) > 0)
        # and the closed-form KL stays numerically stable. 
        self.fc21 = nn.Linear(self.hidden_dim, z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, z_dim)

        # decoder: 4 to 8 to 16 to 32 to 64 dim
        self.fc3 = nn.Linear(z_dim, self.hidden_dim)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)            # 4  to 8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)             # 8  to 16
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)             # 16 to 32
        self.deconv4 = nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1)    # 32 to 64

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = h.view(h.size(0), 128, self.dim_after_conv, self.dim_after_conv)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        return torch.sigmoid(self.deconv4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. "Auto-Encoding Variational Bayes", ICLR 2013
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar).item()
    val_loss /= len(val_loader.dataset)
    print('angstrom Epoch: {} Validation loss: {:.4f}'.format(epoch, val_loss))
    return val_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                # save a few input vs reconstruction comparisons (channel 0
                # = Calpha-Calpha distance map) - protein-equivalent of the
                # pytorch example's torchvision `save_image` call
                n = min(data.size(0), 8)
                save_recon_grid(data[:n], recon_batch[:n],
                                Path(args.outdir) / f'reconstruction_{epoch}.png')

    test_loss /= len(test_loader.dataset)
    print('angstrom Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def save_recon_grid(inputs, recons, path, channel=0):
    """Save an input vs reconstruction grid for a single feature channel.
    Drop-in replacement for torchvision.utils.save_image used in the
    pytorch example, since our 7-channel inputs aren't natively viewable."""
    n = inputs.size(0)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for j in range(n):
        axes[0, j].imshow(inputs[j, channel].cpu().numpy(), vmin=0, vmax=1)
        axes[0, j].set_title("input")
        axes[0, j].axis("off")
        axes[1, j].imshow(recons[j, channel].cpu().numpy(), vmin=0, vmax=1)
        axes[1, j].set_title("recon")
        axes[1, j].axis("off")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    if train_loader is None:
        # No --feature-dir == run the random-tensor smoke test only,
        # based on `vae(torch.rand(10, 1, 32, 32))` in Tutorial 10.
        x_dummy = torch.rand(10, N_CHANNELS, CROP_SIZE, CROP_SIZE).to(device)
        x_recon, mu, logvar = model(x_dummy)
        print("input :", x_dummy.shape)        # (10, 7, 64, 64)
        print("recon :", x_recon.shape)        # (10, 7, 64, 64)
        print("mu    :", mu.shape)             # (10, z_dim)
        print("logvar:", logvar.shape)         # (10, z_dim)
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "z_dim": args.z_dim,
                    "in_channels": N_CHANNELS,
                    "crop_size": CROP_SIZE,
                    "epoch": epoch,
                },
                outdir / "vae_best.pt",
            )

        with torch.no_grad():
            sample = torch.randn(64, args.z_dim).to(device)
            sample = model.decode(sample).cpu()
            np.save(outdir / f'sample_{epoch}.npy', sample.numpy())

    # Final test-set evaluation on the best-val checkpoint
    ckpt = torch.load(outdir / "vae_best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss = test(args.epochs)

    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    (outdir / "test_metrics.json").write_text(
        json.dumps({"test_loss": test_loss, "best_val": best_val}, indent=2)
    )
