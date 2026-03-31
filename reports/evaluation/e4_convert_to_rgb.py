"""
e4_convert_to_rgb.py

This script converts RGBA images to RGB format for evaluation.

It is primarily used to prepare generated and real images for metrics such as
FID, which require 3-channel RGB inputs. The script processes all PNG images in
a directory and saves the converted results to a new location.
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def convert_directory(input_dir, output_dir, overwrite=False):
    """
    Convert all PNG images in a directory from RGBA to RGB format.

    Args:
        input_dir (Path): Directory containing input images.
        output_dir (Path): Directory to save converted images.
        overwrite (bool): Whether to overwrite existing files.

    Notes:
        - Existing files are skipped unless overwrite=True.
        - Image content is preserved except for channel conversion.
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

        # convert RGBA → RGB (required for FID computation)
        img = Image.open(src_path).convert("RGB")
        img.save(dst_path)

        count += 1

    print(f"Converted {count} images → {output_dir}")


def main():
    """
    Command-line interface for batch image conversion.

    Arguments:
        --input_dir: Directory containing input images.
        --output_dir: Optional output directory (auto-generated if not provided).
        --overwrite: Overwrite existing files if set.
    """

    parser = argparse.ArgumentParser()

    # input directory
    parser.add_argument("--input_dir", type=str, required=True)

    # optional output directory
    parser.add_argument("--output_dir", type=str, default=None)

    # overwrite option
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # auto-create output directory if not provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_rgb"

    convert_directory(input_dir, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()