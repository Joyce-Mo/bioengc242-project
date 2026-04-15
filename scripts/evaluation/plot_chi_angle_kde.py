#!/usr/bin/env python
"""Plot KDE distributions of side-chain chi angles from PDB files.

Reproduces the rotamer analysis from La-Proteina (Appendix D.3.2):
KDE plots for all side-chain angles of all amino acids, comparing
generated structures against a reference PDB dataset (e.g., CATH-20).

Usage:
    python scripts/plot_chi_angle_kde.py \
        --input_dirs "Protpardelle=/path/to/generated" "CATH-20=/path/to/cath" \
        --output_dir outputs/chi_angle_plots

Each --input_dirs entry is NAME=PATH where PATH contains .pdb files.
The reference dataset should be experimental structures (e.g., Ingraham CATH
or a filtered PDB subset), NOT MPNN/ESMFold-generated structures.
"""

import argparse
import glob
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from scipy.stats import gaussian_kde

# Chi angle definitions 
# Format: residue 3-letter code -> list of (chi_name, [atom1, atom2, atom3, atom4])
# based on alphafold: 
# https://github.com/google-deepmind/alphafold/blob/main/alphafold/common/residue_constants.py
CHI_ANGLES = {
    "ARG": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD"]),
        ("chi3", ["CB", "CG", "CD", "NE"]),
        ("chi4", ["CG", "CD", "NE", "CZ"]),
    ],
    "ASN": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "OD1"]),
    ],
    "ASP": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "OD1"]),
    ],
    "CYS": [("chi1", ["N", "CA", "CB", "SG"])],
    "GLN": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD"]),
        ("chi3", ["CB", "CG", "CD", "OE1"]),
    ],
    "GLU": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD"]),
        ("chi3", ["CB", "CG", "CD", "OE1"]),
    ],
    "HIS": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "ND1"]),
    ],
    "ILE": [
        ("chi1", ["N", "CA", "CB", "CG1"]),
        ("chi2", ["CA", "CB", "CG1", "CD1"]),
    ],
    "LEU": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD1"]),
    ],
    "LYS": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD"]),
        ("chi3", ["CB", "CG", "CD", "CE"]),
        ("chi4", ["CG", "CD", "CE", "NZ"]),
    ],
    "MET": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "SD"]),
        ("chi3", ["CB", "CG", "SD", "CE"]),
    ],
    "PHE": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD1"]),
    ],
    "PRO": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD"]),
    ],
    "SER": [("chi1", ["N", "CA", "CB", "OG"])],
    "THR": [("chi1", ["N", "CA", "CB", "OG1"])],
    "TRP": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD1"]),
    ],
    "TYR": [
        ("chi1", ["N", "CA", "CB", "CG"]),
        ("chi2", ["CA", "CB", "CG", "CD1"]),
    ],
    "VAL": [("chi1", ["N", "CA", "CB", "CG1"])],
}

AA_WITH_CHI = sorted(CHI_ANGLES.keys())

# Reference dataset names — these get solid black lines and are plotted first.
# Any input name containing one of these substrings (case-insensitive) is treated as reference.
REFERENCE_KEYWORDS = {"cath", "pdb", "afdb"}

_MODEL_COLORS = [
    "#5B8DB8",  # steel blue
    "#E07B54",  # warm coral
    "#B07AA1",  # muted purple
    "#F2C14E",  # golden yellow
    "#76B7B2",  # teal
    "#E15759",  # rose red
    "#59A14F",  # forest green
    "#FF9DA7",  # soft pink
]


def _is_reference(name):
    """Check if a method name is a reference dataset."""
    lower = name.lower()
    return any(kw in lower for kw in REFERENCE_KEYWORDS)


def assign_styles(method_names):
    """Auto-assign colors and linestyles to method names.

    Reference datasets get solid black lines. Generated models get
    distinct colors with dashed lines. Order: references first, then
    models in input order.

    Returns:
        dict: {name: {"color": str, "linestyle": str, "linewidth": float}}
    """
    styles = {}
    model_idx = 0
    for name in method_names:
        if _is_reference(name):
            styles[name] = {
                "color": "#333333",
                "linestyle": "-",
                "linewidth": 2.0,
            }
        else:
            styles[name] = {
                "color": _MODEL_COLORS[model_idx % len(_MODEL_COLORS)],
                "linestyle": "--",
                "linewidth": 1.5,
            }
            model_idx += 1
    return styles

# Geometry

def dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle in degrees from four 3D points (numpy arrays)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-7 or n2_norm < 1e-7:
        return None
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(-y, x))


# Extract chi angles from PDB files 

def extract_chi_angles_from_pdb(pdb_path):
    """Extract all chi angles from a PDB file.

    Returns:
        dict: {(resname, chi_name): [angle_in_degrees, ...]}
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", pdb_path)
    except Exception:
        return {}

    angles = defaultdict(list)

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname not in CHI_ANGLES:
                    continue

                for chi_name, atom_names in CHI_ANGLES[resname]:
                    coords = []
                    missing = False
                    for aname in atom_names:
                        if aname in residue:
                            coords.append(residue[aname].get_vector().get_array())
                        else:
                            missing = True
                            break
                    if missing:
                        continue

                    angle = dihedral_angle(*[np.array(c) for c in coords])
                    if angle is not None:
                        angles[(resname, chi_name)].append(angle)

    return angles


def extract_chi_angles_from_dir(pdb_dir, max_files=None):
    """Extract chi angles from all PDB files in a directory.

    Returns:
        dict: {(resname, chi_name): np.array of angles in degrees}
    """
    pdb_files = sorted(
        glob.glob(os.path.join(pdb_dir, "**/*.pdb"), recursive=True)
    )
    if not pdb_files:
        pdb_files = sorted(
            glob.glob(os.path.join(pdb_dir, "**/*.cif"), recursive=True)
        )
    if max_files is not None:
        pdb_files = pdb_files[:max_files]

    print(f"  Found {len(pdb_files)} PDB files in {pdb_dir}")

    all_angles = defaultdict(list)
    for i, pdb_file in enumerate(pdb_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(pdb_files)}...")
        file_angles = extract_chi_angles_from_pdb(pdb_file)
        for key, vals in file_angles.items():
            all_angles[key].extend(vals)

    return {k: np.array(v) for k, v in all_angles.items()}


# Plotting 

def plot_kde_for_amino_acid(
    resname,
    all_data,
    output_path,
    styles,
    bw_method=None, # default scott rule n**(-1./(d+4)),
    figsize_per_plot=(6, 3.5),
):
    """Plot KDE for all chi angles of one amino acid.

    Args:
        resname: 3-letter amino acid code
        all_data: dict of {method_name: {(resname, chi_name): np.array}}
        output_path: path to save the figure
        styles: dict from assign_styles()
        bw_method: KDE bandwidth (scalar or string)
        figsize_per_plot: (width, height) per subplot
    """
    chi_defs = CHI_ANGLES[resname]
    n_chi = len(chi_defs)

    ncols = min(n_chi, 3)
    nrows = (n_chi + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for idx, (chi_name, _) in enumerate(chi_defs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        key = (resname, chi_name)

        chi_num = int(chi_name.replace("chi", ""))
        x_min, x_max = -180, 180
        x_plot = np.linspace(x_min, x_max, 500)

        # Plot references first, then generated models
        plot_order = sorted(all_data.keys(), key=lambda n: (not _is_reference(n), n))
        for method_name in plot_order:
            method_data = all_data[method_name]
            if key not in method_data or len(method_data[key]) < 10:
                continue

            angles = method_data[key]
            s = styles[method_name]

            try:
                kde = gaussian_kde(angles, bw_method=bw_method)
                density = kde(x_plot)
                ax.plot(
                    x_plot, density,
                    label=method_name,
                    color=s["color"],
                    linestyle=s["linestyle"],
                    linewidth=s["linewidth"],
                    alpha=0.85,
                )
            except Exception:
                continue

        chi_label = f"\u03c7{chi_num}"
        ax.set_xlabel(f"{resname} {chi_label} Angle (degrees)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=9)

    for idx in range(n_chi, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )

    fig.suptitle(
        f"Side-chain angles for amino acid {resname}",
        fontsize=12,
        fontweight="bold",
        y=-0.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KDE of side-chain chi angles from PDB files."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="NAME=PATH pairs, e.g. Protpardelle=/path/to/pdbs CATH-20=/path/to/cath",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/chi_angle_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max PDB files to load per method (for quick testing)",
    )
    parser.add_argument(
        "--bw_method",
        type=float,
        default=None,
        help="KDE bandwidth parameter (default: scipy Scott's rule)",
    )
    parser.add_argument(
        "--amino_acids",
        nargs="*",
        default=None,
        help="Specific amino acids to plot (3-letter codes). Default: all with chi angles.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse input directories
    methods = {}
    for entry in args.input_dirs:
        if "=" not in entry:
            raise ValueError(
                f"Expected NAME=PATH format, got: {entry}"
            )
        name, path = entry.split("=", 1)
        methods[name] = path

    # Extract chi angles for each method
    all_data = {}
    for name, path in methods.items():
        print(f"Extracting chi angles for {name}...")
        all_data[name] = extract_chi_angles_from_dir(path, max_files=args.max_files)
        total = sum(len(v) for v in all_data[name].values())
        print(f"  Total angles extracted: {total}")

    # Determine which amino acids to plot
    aa_list = args.amino_acids if args.amino_acids else AA_WITH_CHI

    # Auto-assign colors and linestyles
    styles = assign_styles(list(all_data.keys()))
    print("\nStyle assignments:")
    for name, s in styles.items():
        role = "reference" if _is_reference(name) else "model"
        print(f"  {name} ({role}): {s['color']}, {s['linestyle']}")

    # Generate plots
    print("\nGenerating KDE plots...")
    for resname in aa_list:
        if resname not in CHI_ANGLES:
            print(f"  Skipping {resname} (no chi angles defined)")
            continue
        output_path = os.path.join(
            args.output_dir, f"chi_angles_{resname}.{args.format}"
        )
        plot_kde_for_amino_acid(
            resname, all_data, output_path, styles, bw_method=args.bw_method
        )

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
