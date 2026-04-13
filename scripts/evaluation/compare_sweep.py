#!/usr/bin/env python
"""Compare hyperparameter sweep results and identify best config.

Reads test_metrics.json and hparams.json from each sweep run directory,
ranks by test loss, and produces a summary table + training curve plots.

Usage:
    python scripts/evaluation/compare_sweep.py /path/to/vae_sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_run(run_dir):
    """Load metrics and hparams from a single run directory."""
    metrics_path = run_dir / "test_metrics.json"
    hparams_path = run_dir / "hparams.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    hparams = json.loads(hparams_path.read_text()) if hparams_path.exists() else {}
    return {
        "run": run_dir.name,
        "test_loss": metrics.get("test_loss"),
        "best_val": metrics.get("best_val"),
        "best_epoch": metrics.get("best_epoch"),
        "total_epochs": metrics.get("total_epochs"),
        "z_dim": hparams.get("z_dim"),
        "lr": hparams.get("lr"),
        "dropout": hparams.get("dropout"),
        "weight_decay": hparams.get("weight_decay"),
        "kl_anneal_epochs": hparams.get("kl_anneal_epochs"),
        "use_batchnorm": hparams.get("use_batchnorm"),
        "lr_schedule": hparams.get("lr_schedule"),
    }


def plot_training_curves(sweep_dir, runs_df, output_path):
    """Plot val loss curves for top runs."""
    top_runs = runs_df.nsmallest(6, "test_loss")
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in top_runs.iterrows():
        history_path = sweep_dir / row["run"] / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        ax.plot(history["val"], label=row["run"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Top 6 Configs — Validation Loss")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare VAE sweep results")
    parser.add_argument("sweep_dir", type=str, help="Directory containing sweep run subdirs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <sweep_dir>/sweep_summary.csv)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        result = load_run(run_dir)
        if result is not None:
            runs.append(result)

    if not runs:
        print(f"No completed runs found in {sweep_dir}")
        return

    df = pd.DataFrame(runs).sort_values("test_loss")

    csv_path = args.output or str(sweep_dir / "sweep_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP RESULTS (ranked by test loss)")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nBest config: {df.iloc[0]['run']}")
    print(f"  test_loss={df.iloc[0]['test_loss']:.4f}  best_val={df.iloc[0]['best_val']:.4f}")
    print(f"\nSaved to {csv_path}")

    plot_training_curves(sweep_dir, df, sweep_dir / "sweep_val_curves.png")
    print(f"Training curves saved to {sweep_dir / 'sweep_val_curves.png'}")


if __name__ == "__main__":
    main()
