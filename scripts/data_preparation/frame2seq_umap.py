"""Generate UMAP representations of Frame2seq layer embeddings.

Extracts per-residue embeddings from each of the 8 IPA layers in
Frame2seq and produces UMAP visualizations colored by predicted amino
acid identity, similar to Figure S1 in:
  "Structure-conditioned masked language models for protein sequence
   design generalize beyond the native sequence space"

For each protein, the single representation (s) at each layer is kept
at per-residue resolution. The model's predicted amino acid at each
position is used to color the UMAP points.

Usage:
    python scripts/frame2seq_umap.py --input-dir PATH [--output-dir PATH] [--chain-id A]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

from frame2seq.model.Frame2seq import frame2seq
from frame2seq.utils.pdb2input import get_inference_inputs

# Amino acid ordering and color palette
AA_ORDER = list("DEKRHQNSTPGAVILMCFWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# 20 distinct colors for amino acids (categorical palette)
AA_COLORS = {
    "D": "#E6194B",  # red
    "E": "#F58231",  # orange
    "K": "#3CB44B",  # green
    "R": "#4363D8",  # blue
    "H": "#911EB4",  # purple
    "Q": "#42D4F4",  # cyan
    "N": "#F032E6",  # magenta
    "S": "#BFEF45",  # lime
    "T": "#FABED4",  # pink
    "P": "#469990",  # teal
    "G": "#DCBEFF",  # lavender
    "A": "#9A6324",  # brown
    "V": "#FFFAC8",  # beige
    "I": "#800000",  # maroon
    "L": "#AAFFC3",  # mint
    "M": "#808000",  # olive
    "C": "#FFD8B1",  # apricot
    "F": "#000075",  # navy
    "W": "#A9A9A9",  # grey
    "Y": "#FFE119",  # yellow
}

# Frame2seq uses 20 standard AA classes; map model output index to one-letter code
# Frame2seq aatype order: A R N D C Q E G H I L K M F P S T W Y V
FRAME2SEQ_IDX_TO_AA = list("ARNDCQEGHILKMFPSTWYV")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "/wynton/home/rotation/jqmo/rotation3/datasets/cath20/cath20-filtered-foldseek"
DEFAULT_OUTPUT = "/wynton/home/rotation/jqmo/rotation3/protein_augmentation/output/frame2seq_umap"

SUPPORTED_EXTENSIONS = {".pdb"}


def extract_layer_embeddings(model, pdb_file, chain_id, device):
    """Run a forward pass and capture per-residue single representations at each layer.

    Parameters
    ----------
    model : frame2seq
        A loaded Frame2seq model in eval mode.
    pdb_file : str
        Path to PDB file.
    chain_id : str
        Chain identifier.
    device : torch.device
        Device for inference.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        - layer_embeddings: list of length ipa_depth, each entry is an array
          of shape (num_residues, embed_dim) with per-residue embeddings.
        - predicted_aa: array of shape (num_residues,) with predicted amino
          acid indices (into FRAME2SEQ_IDX_TO_AA).
    """
    from frame2seq.utils.rigid_utils import Rigid
    from frame2seq.utils.featurize import make_s_init, make_z_init

    seq_mask, aatype, X = get_inference_inputs(pdb_file, chain_id)

    # Prepare masked input (all positions masked for unconditional embedding)
    input_S = torch.zeros(1, aatype.shape[1], model.sequence_dim).to(device)

    X = X.to(device)
    seq_mask = seq_mask.to(device)
    input_S = input_S.to(device)

    with torch.no_grad():
        r = Rigid.from_3_points(X[:, :, 0], X[:, :, 1], X[:, :, 2])
        s, in_S = make_s_init(model, X, input_S, seq_mask)
        s = model.sequence_to_single(s)
        s = s + model.input_sequence_layer_norm(in_S)
        z = make_z_init(model, X)
        z = model.edge_to_pair(z)
        seq_mask_long = seq_mask.long()

        layer_embeddings = []

        for ipa, ipa_dropout, layer_norm_ipa, *transit_layers, edge_transition in model.layers:
            s = s + ipa(s, z, r, seq_mask_long, attn_drop_rate=0.0)
            s = layer_norm_ipa(s)

            if model.st_mod_tsit_factor > 1:
                pre_transit = transit_layers[0]
                transition = transit_layers[1]
                post_transit = transit_layers[2]
                s = pre_transit(s)
                s = transition(s)
                s = post_transit(s)
            else:
                transition = transit_layers[0]
                s = transition(s)

            if edge_transition is not None:
                z = edge_transition(s, z)

            # Keep per-residue embeddings (masked positions only)
            mask = seq_mask.squeeze(0).bool()
            per_residue = s[0, mask].cpu().numpy()  # (num_residues, embed_dim)
            layer_embeddings.append(per_residue)

        # Get predicted amino acid from final layer
        logits = model.single_to_sequence(s)  # (1, seq_len, 20)
        mask = seq_mask.squeeze(0).bool()
        predicted_aa = logits[0, mask].argmax(dim=-1).cpu().numpy()  # (num_residues,)

    return layer_embeddings, predicted_aa


def load_models(device):
    """Load Frame2seq model ensemble.

    Returns
    -------
    list[frame2seq]
        List of loaded models.
    """
    import os
    from glob import glob

    frame2seq_path = os.path.dirname(os.path.abspath(
        __import__('frame2seq').__file__
    ))
    trained_models_dir = os.path.join(frame2seq_path, 'trained_models')
    model_ckpts = glob(os.path.join(trained_models_dir, '*.ckpt'))

    if not model_ckpts:
        logger.error("No model checkpoints found in %s", trained_models_dir)
        sys.exit(1)

    models = []
    for ckpt_file in model_ckpts:
        logger.info("Loading %s", ckpt_file)
        model = frame2seq.load_from_checkpoint(ckpt_file).eval().to(device)
        models.append(model)

    return models


def compute_embeddings(models, pdb_files, chain_id, device):
    """Extract per-residue layer embeddings for all PDB files, averaged across model ensemble.

    Parameters
    ----------
    models : list[frame2seq]
        Loaded model ensemble.
    pdb_files : list[Path]
        PDB files to process.
    chain_id : str
        Chain ID to use.
    device : torch.device
        Compute device.

    Returns
    -------
    dict
        'embeddings': list of length num_layers, each np.ndarray of shape
            (total_residues, embed_dim)
        'aa_labels': np.ndarray of shape (total_residues,) with one-letter AA codes
        'names': list of protein names
        'num_layers': int
    """
    # Collect per-layer lists of per-residue embeddings
    per_layer_embs = None  # will be list of lists once num_layers is known
    all_aa_labels = []
    names = []

    for i, pdb_file in enumerate(pdb_files):
        logger.info("Processing %d/%d: %s", i + 1, len(pdb_files), pdb_file.name)

        try:
            # Average embeddings across ensemble
            ensemble_layer_embs = []
            ensemble_aa_preds = []
            for model in models:
                layer_embs, predicted_aa = extract_layer_embeddings(
                    model, str(pdb_file), chain_id, device
                )
                ensemble_layer_embs.append(layer_embs)  # list of (num_res, embed_dim)
                ensemble_aa_preds.append(predicted_aa)   # (num_res,)

            num_layers = len(ensemble_layer_embs[0])
            if per_layer_embs is None:
                per_layer_embs = [[] for _ in range(num_layers)]

            # Average per-residue embeddings across ensemble at each layer
            for layer_idx in range(num_layers):
                layer_stack = np.stack(
                    [e[layer_idx] for e in ensemble_layer_embs]
                )  # (num_models, num_res, embed_dim)
                avg_layer = layer_stack.mean(axis=0)  # (num_res, embed_dim)
                per_layer_embs[layer_idx].append(avg_layer)

            # Use majority-vote predicted AA across ensemble
            aa_stack = np.stack(ensemble_aa_preds)  # (num_models, num_res)
            from scipy.stats import mode
            majority_aa, _ = mode(aa_stack, axis=0)
            majority_aa = majority_aa.flatten()

            # Convert model indices to one-letter codes
            aa_letters = np.array([FRAME2SEQ_IDX_TO_AA[idx] for idx in majority_aa])
            all_aa_labels.append(aa_letters)
            names.append(pdb_file.stem)
        except Exception as e:
            logger.warning("Failed on %s: %s", pdb_file.name, e)
            continue

    if per_layer_embs is None or not per_layer_embs[0]:
        logger.error("No embeddings extracted.")
        sys.exit(1)

    num_layers = len(per_layer_embs)

    # Concatenate all residues for each layer
    embeddings_by_layer = [
        np.concatenate(per_layer_embs[l], axis=0) for l in range(num_layers)
    ]
    aa_labels = np.concatenate(all_aa_labels)

    return {
        "embeddings": embeddings_by_layer,
        "aa_labels": aa_labels,
        "names": names,
        "num_layers": num_layers,
    }


def _get_aa_colors(aa_labels):
    """Map amino acid labels to colors.

    Parameters
    ----------
    aa_labels : np.ndarray
        Array of one-letter amino acid codes.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    return [AA_COLORS.get(aa, "#808080") for aa in aa_labels]


def _aa_legend(ax):
    """Add a legend mapping amino acid letters to colors."""
    patches = [
        mpatches.Patch(color=AA_COLORS[aa], label=aa)
        for aa in AA_ORDER
    ]
    ax.legend(
        handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5),
        ncol=1, fontsize=7, frameon=False, handlelength=1, handletextpad=0.4,
    )


def plot_umap_grid(embeddings_dict, output_dir):
    """Generate a grid of UMAP plots colored by predicted amino acid, one per layer.

    Parameters
    ----------
    embeddings_dict : dict
        Output from compute_embeddings().
    output_dir : Path
        Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embeddings_dict["embeddings"]
    aa_labels = embeddings_dict["aa_labels"]
    names = embeddings_dict["names"]
    num_layers = embeddings_dict["num_layers"]

    colors = _get_aa_colors(aa_labels)

    # Grid layout: 2 rows x 4 cols for 8 layers
    ncols = 4
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_embs = embeddings[layer_idx]  # (total_residues, embed_dim)

        reducer = umap.UMAP(n_neighbors=100, min_dist=0.3, random_state=42, metric='euclidean')
        umap_coords = reducer.fit_transform(layer_embs)

        # Shuffle plot order so no single AA dominates the foreground
        order = np.random.RandomState(42).permutation(len(umap_coords))
        ax.scatter(
            umap_coords[order, 0], umap_coords[order, 1],
            c=[colors[i] for i in order],
            s=5, alpha=0.5, edgecolors='none',
        )
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    _aa_legend(axes[num_layers - 1])

    fig.suptitle("Frame2seq Layer Embeddings (UMAP) — colored by predicted AA", fontsize=16, y=1.02)
    plt.tight_layout()

    grid_path = output_dir / "umap_layers_grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grid plot to %s", grid_path)

    # Also save individual layer plots
    for layer_idx in range(num_layers):
        layer_embs = embeddings[layer_idx]
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_coords = reducer.fit_transform(layer_embs)

        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        order = np.random.RandomState(42).permutation(len(umap_coords))
        ax_ind.scatter(
            umap_coords[order, 0], umap_coords[order, 1],
            c=[colors[i] for i in order],
            s=10, alpha=0.6, edgecolors='none',
        )
        ax_ind.set_title(f"Frame2seq Layer {layer_idx + 1} Embeddings (UMAP)", fontsize=14)
        ax_ind.set_xlabel("UMAP 1")
        ax_ind.set_ylabel("UMAP 2")
        _aa_legend(ax_ind)

        ind_path = output_dir / f"umap_layer_{layer_idx + 1}.png"
        fig_ind.savefig(ind_path, dpi=200, bbox_inches="tight")
        plt.close(fig_ind)

    logger.info("Saved %d individual layer plots to %s", num_layers, output_dir)

    # Save raw embeddings, AA labels, and UMAP coordinates
    np.savez(
        output_dir / "layer_embeddings.npz",
        **{f"layer_{i}": embeddings[i] for i in range(num_layers)},
        aa_labels=aa_labels,
        names=np.array(names),
    )
    logger.info("Saved raw embeddings to %s", output_dir / "layer_embeddings.npz")


def main():
    """Entry point for Frame2seq UMAP embedding script."""
    parser = argparse.ArgumentParser(
        description="Generate UMAP visualizations of Frame2seq layer embeddings.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Directory of PDB files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for plots and embeddings (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default="A",
        help="Chain ID to use (default: A)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        logger.error("No PDB files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d PDB files", len(pdb_files))

    models = load_models(device)
    logger.info("Loaded %d model(s)", len(models))

    embeddings_dict = compute_embeddings(models, pdb_files, args.chain_id, device)
    plot_umap_grid(embeddings_dict, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
