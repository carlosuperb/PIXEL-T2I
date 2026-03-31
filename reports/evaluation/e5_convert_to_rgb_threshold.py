"""
e5_convert_to_rgb_threshold.py

This script converts RGBA images to RGB format using alpha thresholding.

Instead of directly dropping the alpha channel, a threshold is applied to remove
low-opacity pixels, reducing background artefacts. This produces more consistent
RGB images for evaluation metrics such as FID.
"""

import os
from pathlib import Path
from PIL import Image
import argparse
import numpy as np


def rgba_to_rgb_with_threshold(img, threshold=128):
    """
    Convert an RGBA image to RGB using alpha thresholding.

    Pixels with alpha values below the threshold are treated as background and
    removed, while remaining pixels are preserved.

    Args:
        img (PIL.Image): Input image.
        threshold (int): Alpha threshold (0-255).

    Returns:
        PIL.Image: RGB image with thresholded background removed.
    """

    # ensure RGBA format
    img = img.convert("RGBA")
    rgba = np.array(img)

    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    # apply alpha threshold (foreground mask)
    alpha_bin = (alpha > threshold).astype(np.uint8)

    # expand mask for RGB channels
    alpha_bin = alpha_bin[..., None]

    # composite onto black background
    rgb_out = rgb * alpha_bin

    return Image.fromarray(rgb_out.astype(np.uint8))


def convert_directory(input_dir, output_dir, overwrite=False, threshold=128):
    """
    Convert all PNG images in a directory using alpha-threshold-based RGBA→RGB conversion.

    Args:
        input_dir (Path): Directory containing input images.
        output_dir (Path): Directory to save converted images.
        overwrite (bool): Whether to overwrite existing files.
        threshold (int): Alpha threshold for foreground extraction.

    Notes:
        - Files are skipped if they already exist unless overwrite=True.
        - This method reduces noise from semi-transparent pixels, improving
          evaluation stability for FID.
    """

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # list all PNG files
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    print(f"Found {len(files)} images in {input_dir}")

    count = 0

    for fname in files:
        src_path = input_dir / fname
        dst_path = output_dir / fname

        # skip if already exists
        if dst_path.exists() and not overwrite:
            continue

        try:
            img = Image.open(src_path)

            # convert RGBA → RGB with thresholding
            img = rgba_to_rgb_with_threshold(img, threshold=threshold)

            img.save(dst_path)
            count += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Converted {count} images → {output_dir}")


def main():
    """
    Command-line interface for batch RGBA to RGB conversion with thresholding.

    Arguments:
        --input_dir: Directory containing input images.
        --output_dir: Optional output directory (auto-generated if not provided).
        --overwrite: Overwrite existing files if set.
        --threshold: Alpha threshold (default: 128).
    """

    parser = argparse.ArgumentParser()

    # input directory
    parser.add_argument("--input_dir", type=str, required=True)

    # optional output directory
    parser.add_argument("--output_dir", type=str, default=None)

    # overwrite option
    parser.add_argument("--overwrite", action="store_true")

    # threshold option
    parser.add_argument("--threshold", type=int, default=128)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # auto-create output directory if not provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_rgb"

    convert_directory(
        input_dir,
        output_dir,
        overwrite=args.overwrite,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()