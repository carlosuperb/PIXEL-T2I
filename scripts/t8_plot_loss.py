"""
t8_plot_loss.py

Plot training loss curves from a CSV log file.
All output metadata (title and save path) must be explicitly specified
via command-line arguments to ensure reproducibility.

Usage (ALL arguments required):
    python scripts/t8_plot_loss.py \
      --csv outputs/logs_unconditional/loss_history.csv \
      --output reports/figures/training_curves/unconditional_loss_curve.png \
      --title "Training Loss Curve (Unconditional)"
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_loss_curve(csv_path: str, save_path: str, title: str) -> None:
    """
    Plot smooth training loss curve and save to disk.

    Args:
        csv_path: Path to loss_history.csv (required)
        save_path: Path to save plot image (required)
        title: Plot title (required)
    """
    # read CSV
    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns or "loss" not in df.columns:
        raise ValueError("CSV must contain columns: 'epoch' and 'loss'.")

    if len(df) < 1:
        raise ValueError("CSV is empty (no rows).")

    print("Training data loaded:")
    print(f"  Total epochs: {len(df)}")
    print(f"  Initial loss: {df['loss'].iloc[0]:.6f}")
    print(f"  Final loss: {df['loss'].iloc[-1]:.6f}")
    print(f"  Best loss: {df['loss'].min():.6f} at epoch {df['loss'].idxmin() + 1}")
    print(
        f"  Loss reduction: "
        f"{(df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100:.1f}%"
    )

    # data
    epochs = df["epoch"].to_numpy()
    loss = df["loss"].to_numpy()

    # interpolate (smooth)
    if len(epochs) > 3:  # Need at least 4 points for cubic spline
        epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        spl = make_interp_spline(epochs, loss, k=3)
        loss_smooth = spl(epochs_smooth)
    else:
        epochs_smooth = epochs
        loss_smooth = loss

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs_smooth, loss_smooth, linewidth=2.5, label="Training Loss")
    ax.scatter(epochs, loss, s=80, zorder=5, alpha=0.6)

    # Mark best loss
    best_idx = int(df["loss"].idxmin())
    best_epoch = df["epoch"].iloc[best_idx]
    best_loss = df["loss"].iloc[best_idx]
    ax.scatter(
        [best_epoch],
        [best_loss],
        s=200,
        zorder=10,
        label=f"Best: {best_loss:.4f} @ epoch {best_epoch}",
        edgecolors="black",
        linewidths=2,
        marker="o",
    )

    # Styling
    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Loss", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Integer ticks for epochs
    ax.set_xticks(epochs)

    plt.tight_layout()

    # save (required)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curve (all args required)")
    parser.add_argument("--csv", type=str, required=True, help="Path to loss_history.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to save output image")
    parser.add_argument("--title", type=str, required=True, help="Title of the plot")

    args = parser.parse_args()

    plot_loss_curve(
        csv_path=args.csv,
        save_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
