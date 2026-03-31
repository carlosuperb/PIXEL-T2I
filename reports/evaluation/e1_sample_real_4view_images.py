"""
e1_sample_real_4view_images.py

This script randomly samples a subset of real 4-view character images from the
processed dataset and copies them into a dedicated evaluation directory.

The sampled images are used as the reference "real" distribution when computing
evaluation metrics such as FID, ensuring a consistent and reproducible setup.
"""

import os
import random
import shutil
from pathlib import Path

# source dataset (processed 4-view images)
SOURCE_DIR = Path("pixel_character_dataset/processed/dataset_4view/images")

# target directory for evaluation (real images)
TARGET_DIR = Path("reports/evaluation/real_4view")

# sampling configuration
NUM_SAMPLES = 1000
SEED = 42


def main():
    """
    Sample a fixed number of images from the dataset and copy them to the
    evaluation directory.

    The sampling is performed without replacement and controlled by a fixed
    random seed to ensure reproducibility across evaluation runs.

    Raises:
        ValueError: if the dataset contains fewer images than NUM_SAMPLES.
    """

    # ensure deterministic sampling
    random.seed(SEED)

    # create output directory if it does not exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # collect all PNG images in the dataset
    all_images = sorted([
        f for f in os.listdir(SOURCE_DIR)
        if f.endswith(".png")
    ])

    print(f"Total images found: {len(all_images)}")

    # check dataset size is sufficient
    if len(all_images) < NUM_SAMPLES:
        raise ValueError("Not enough images in dataset")

    # randomly select images (no replacement)
    selected = random.sample(all_images, NUM_SAMPLES)

    print(f"Sampling {NUM_SAMPLES} images...")

    # copy selected images to evaluation directory
    for fname in selected:
        src = SOURCE_DIR / fname
        dst = TARGET_DIR / fname
        shutil.copy(src, dst)

    print(f"Done! Saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()