"""
v0_build_release_weights.py

Build a clean release zip for pretrained model weights.

This script:
- Copies only the required .pt files
- Preserves the original directory structure under models/
- Produces a release-ready zip suitable for GitHub Releases
- Cleans up all temporary files after packaging

Run from VS Code or terminal:
  python scripts/v0_build_release_weights.py --version v0.1.0
"""

from pathlib import Path
import shutil
import zipfile
import argparse


WEIGHTS = [
    (
        Path("models/pixel_image_conditional/checkpoints/char_encoder_best.pt"),
        Path("models/pixel_image_conditional/checkpoints/char_encoder_best.pt"),
    ),
    (
        Path("models/pixel_image_conditional/checkpoints/unet_best.pt"),
        Path("models/pixel_image_conditional/checkpoints/unet_best.pt"),
    ),
    (
        Path("models/pixel_text_conditional/checkpoints/model_best.pt"),
        Path("models/pixel_text_conditional/checkpoints/model_best.pt"),
    ),
    (
        Path("models/pixel_unconditional/checkpoints/model_best.pt"),
        Path("models/pixel_unconditional/checkpoints/model_best.pt"),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Build release zip for pretrained model weights."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version tag (e.g. v0.1.0, v0.2.0)",
    )
    args = parser.parse_args()

    version = args.version
    zip_name = f"pixel_t2i_weights_{version}.zip"

    # scripts/ -> repo root
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / "_weights_tmp"
    zip_path = repo_root / zip_name

    print(f"[info] Repo root: {repo_root}")
    print(f"[info] Release version: {version}")
    print(f"[info] Output zip: {zip_path}")

    # Check all required weight files exist
    for src, _ in WEIGHTS:
        abs_src = repo_root / src
        if not abs_src.exists():
            raise FileNotFoundError(f"Missing weight file: {abs_src}")

    # Remove old zip if present
    if zip_path.exists():
        print(f"[clean] Removing existing zip: {zip_path}")
        zip_path.unlink()

    # Remove old temp directory if present
    if tmp_dir.exists():
        print(f"[clean] Removing existing temp dir: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    # Create temp directory structure and copy weights
    print("[build] Preparing temporary directory...")
    for src, dst in WEIGHTS:
        abs_src = repo_root / src
        abs_dst = tmp_dir / dst

        abs_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_src, abs_dst)
        print(f"  - {dst}")

    # Create zip archive with preserved paths
    print("[zip] Creating archive...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for _, dst in WEIGHTS:
            file_path = tmp_dir / dst
            zf.write(
                file_path,
                arcname=dst.as_posix(),
            )

    # Clean up temporary directory
    print("[clean] Removing temporary directory...")
    shutil.rmtree(tmp_dir)

    print("[done] Release zip created successfully.")
    print("       Do not commit this zip to the repository.")
    print(f"       Upload it to GitHub Releases ({version}).")


if __name__ == "__main__":
    main()
