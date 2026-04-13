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
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Adam weight decay / L2 regularization (default: 0)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout probability in encoder/decoder (default: 0)')
parser.add_argument('--use-batchnorm', action='store_true',
                    help='add BatchNorm2d after each conv/deconv layer')
parser.add_argument('--kl-anneal-epochs', type=int, default=0,
                    help='linearly anneal KL weight from 0 to 1 over N epochs (0=off)')
parser.add_argument('--lr-schedule', choices=['none', 'plateau', 'cosine'], default='none',
                    help='learning rate schedule (default: none)')
parser.add_argument('--lr-patience', type=int, default=10,
                    help='patience for ReduceLROnPlateau (default: 10)')
parser.add_argument('--early-stop-patience', type=int, default=0,
                    help='stop if val loss does not improve for N epochs (0=off)')
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
    def __init__(self, in_channels=N_CHANNELS, z_dim=args.z_dim,
                 dropout=args.dropout, use_batchnorm=args.use_batchnorm):
        super(VAE, self).__init__()
        self.dropout_p = dropout
        self.use_bn = use_batchnorm

        # encoder: 64 to 32 to 16 to 8 to 4
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

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


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# LR scheduler
scheduler = None
if args.lr_schedule == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience)
elif args.lr_schedule == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def kl_weight(epoch):
    """Linear KL annealing: weight goes from 0 to 1 over kl_anneal_epochs."""
    if args.kl_anneal_epochs <= 0:
        return 1.0
    return min(1.0, epoch / args.kl_anneal_epochs)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_w=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. "Auto-Encoding Variational Bayes", ICLR 2013
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + kl_w * KLD


def train(epoch):
    model.train()
    train_loss = 0
    kl_w = kl_weight(epoch)
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, kl_w=kl_w)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} Average loss: {:.4f} (kl_weight={:.3f}, lr={:.2e})'.format(
          epoch, avg_loss, kl_w, optimizer.param_groups[0]['lr']))
    return avg_loss


def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar).item()
    val_loss /= len(val_loader.dataset)
    print('Epoch: {} Validation loss: {:.4f}'.format(epoch, val_loss))
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

    # Save hyperparameters for reproducibility
    hparams = {k: v for k, v in vars(args).items()}
    (outdir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    history = {"train": [], "val": [], "lr": []}
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        # LR scheduling
        if scheduler is not None:
            if args.lr_schedule == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "z_dim": args.z_dim,
                    "in_channels": N_CHANNELS,
                    "crop_size": CROP_SIZE,
                    "epoch": epoch,
                    "hparams": hparams,
                },
                outdir / "vae_best.pt",
            )
        else:
            epochs_no_improve += 1

        with torch.no_grad():
            sample = torch.randn(64, args.z_dim).to(device)
            sample = model.decode(sample).cpu()
            np.save(outdir / f'sample_{epoch}.npy', sample.numpy())

        # Early stopping
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f'Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs)')
            break

    # Final test-set evaluation on the best-val checkpoint
    ckpt = torch.load(outdir / "vae_best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss = test(epoch)

    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    (outdir / "test_metrics.json").write_text(
        json.dumps({"test_loss": test_loss, "best_val": best_val,
                     "best_epoch": ckpt["epoch"], "total_epochs": epoch}, indent=2)
    )
