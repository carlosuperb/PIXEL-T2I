"""
t6_preview_dataset.py

Preview random samples from the 4-view sprite dataset for qualitative inspection.

Features:
  - Randomly sample N images (deterministic with a fixed random seed).
  - For each image, print ORIGINAL vs OPTIMIZED caption to the terminal.
  - Save the ORIGINAL vs OPTIMIZED comparison to a local text file.
  - Create a grid image (near-square layout) of the sampled images and save it
  under reports/preview/.

Input:
  Images directory:
    pixel_character_dataset/processed/dataset_4view/images/

  Original captions:
    pixel_character_dataset/processed/dataset_4view/captions.csv

  Optimized captions:
    pixel_character_dataset/processed/dataset_4view/captions_optimized.csv

Output:
  Figures:
    reports/figures/dataset_previews/dataset_4view_preview.png

  Tables:
    reports/tables/dataset_4view_preview_captions.txt

Console Output:
  - Image filename + [ORIGINAL] caption
  - Image filename + [OPTIMIZED] caption
"""

import csv
import math
import random
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


# ----------------- paths -----------------

SCRIPT_DIR = Path(__file__).resolve().parent      # PIXEL-T2I/scripts
ROOT = SCRIPT_DIR.parent                          # PIXEL-T2I

# Dataset input paths
DATASET_ROOT = ROOT / "pixel_character_dataset" / "processed" / "dataset_4view"
IMG_DIR = DATASET_ROOT / "images"

CAPTION_CSV_ORIG = DATASET_ROOT / "captions.csv"
CAPTION_CSV_OPT  = DATASET_ROOT / "captions_optimized.csv"

# Output preview paths
REPORTS_ROOT = ROOT / "reports"
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

PREVIEW_IMG_PATH  = REPORTS_ROOT / "figures" / "dataset_previews" / "dataset_4view_preview.png"
PREVIEW_TEXT_PATH = REPORTS_ROOT / "tables" / "dataset_4view_preview_captions.txt"


# ----------------- config -----------------

# How many samples to preview (typical range: 1–10)
SAMPLE_COUNT = 6

# Deterministic sampling
RANDOM_SEED = 21

# ----------------- helpers -----------------

def load_captions(path: Path) -> dict[str, str]:
    """
    Read a captions CSV into a dict:
        { "char_00001.png": "caption text", ... }

    Expected CSV columns: image_path, text
    """
    captions: dict[str, str] = {}
    if not path.exists():
        return captions

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image_path", "").strip()
            txt = row.get("text", "").strip()
            if img:
                captions[img] = txt
    return captions


def compute_grid(n: int) -> tuple[int, int]:
    """
    Compute a near-square grid (rows, cols) for n images.

    Examples:
      n=1 -> (1,1)
      n=2 -> (1,2)
      n=4 -> (2,2)
      n=6 -> (2,3)
      n=9 -> (3,3)
      n=10 -> (3,4)
    """
    if n <= 0:
        return 0, 0
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def create_grid_figure(img_names: list[str]) -> None:
    """
    Create a near-square grid figure of the given image filenames
    and save to PREVIEW_IMG_PATH.

    The PNG filename is shown under each image (as x-label).
    """
    n = len(img_names)
    if n == 0:
        print("No images to plot in grid.")
        return

    rows, cols = compute_grid(n)

    # Basic figure size: 3x3 inches per image slot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Normalize axes to a flat list
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for idx, name in enumerate(img_names):
        ax = axes_list[idx]
        img_path = IMG_DIR / name

        if not img_path.exists():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{name} (missing)", fontsize=8)
            continue

        img = Image.open(img_path)
        ax.imshow(img)

        # Remove ticks but keep the axis to show labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the filename below the image
        ax.set_xlabel(name, fontsize=8)

    # Turn off any unused subplots
    for j in range(len(img_names), len(axes_list)):
        ax = axes_list[j]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("", fontsize=8)

    plt.tight_layout()
    fig.savefig(PREVIEW_IMG_PATH, dpi=150)
    plt.close(fig)

    print("Grid preview saved to:", PREVIEW_IMG_PATH)


def preview_samples():
    """
    Preview a small set of randomly sampled 4-view dataset images together with
    their original and optimized captions.

    This function:
    - Deterministically samples N images from the dataset using a fixed random seed.
    - Prints ORIGINAL vs OPTIMIZED captions to the terminal for qualitative inspection.
    - Saves the caption comparison to a text file under reports/tables/.
    - Generates a near-square grid image of the sampled sprites and saves it
      under reports/figures/.

    Intended for quick qualitative sanity checks of dataset quality before
    model training.
    """
    # Basic checks
    if not IMG_DIR.exists():
        print("Error: image directory not found:", IMG_DIR)
        return

    if not CAPTION_CSV_ORIG.exists():
        print("Error: original captions CSV not found:", CAPTION_CSV_ORIG)
        return

    if not CAPTION_CSV_OPT.exists():
        print("Error: optimized captions CSV not found:", CAPTION_CSV_OPT)
        return

    # Load captions
    captions_orig = load_captions(CAPTION_CSV_ORIG)
    captions_opt  = load_captions(CAPTION_CSV_OPT)

    # List all images
    all_imgs = sorted([p.name for p in IMG_DIR.glob("*.png")])

    if len(all_imgs) == 0:
        print("No images found in:", IMG_DIR)
        return

    # Deterministic sampling
    random.seed(RANDOM_SEED)
    k = min(SAMPLE_COUNT, len(all_imgs))
    selected = random.sample(all_imgs, k)

    print(f"\n=== Previewing {k} random samples (of {len(all_imgs)} total) ===\n")

    # Open log file for caption comparison
    with PREVIEW_TEXT_PATH.open("w", encoding="utf-8") as log_f:
        log_f.write(f"Dataset preview: {k} random samples\n")
        log_f.write(f"Images directory: {IMG_DIR}\n")
        log_f.write(f"Original captions: {CAPTION_CSV_ORIG}\n")
        log_f.write(f"Optimized captions: {CAPTION_CSV_OPT}\n")
        log_f.write("=" * 80 + "\n\n")

        # Text preview: original vs optimized
        for name in selected:
            orig = captions_orig.get(name, "(original caption not found)")
            opt  = captions_opt.get(name, "(optimized caption not found)")

            block = (
                f"Image: {name}\n"
                f"Path : {IMG_DIR / name}\n\n"
                "[ORIGINAL]\n"
                f"{orig}\n\n"
                "[OPTIMIZED]\n"
                f"{opt}\n"
                + "-" * 80 + "\n\n"
            )

            # Print to terminal
            print(block, end="")

            # Write to file
            log_f.write(block)

    print(f"\nCaption comparison saved to: {PREVIEW_TEXT_PATH}\n")
    print("Generating grid image preview...\n")

    # Image grid preview
    create_grid_figure(selected)

    print("\n=== Preview complete ===\n")


def main():
    preview_samples()


if __name__ == "__main__":
    main()
