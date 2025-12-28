"""
v1_extract_weights_zip.py

Extract pretrained model weights from a zip archive into the repository root.

Typical workflow:
1) Download the pretrained weights zip from GitHub Releases.
2) Place the zip file in the repository root directory (PIXEL-T2I/).
3) Run:
   python scripts/v1_extract_weights_zip.py

Optional:
- Specify a zip file explicitly:
    python scripts/v1_extract_weights_zip.py --zip pixel_t2i_weights_v0.1.0.zip
- Force extraction even if files already exist:
    python scripts/v1_extract_weights_zip.py --force
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_PREFIX = "models/"


def pick_default_zip(repo_root: Path) -> Path | None:
    """
    Select the most recently modified pretrained weights zip in the repository root.

    The function looks for files matching the pattern:
        pixel_t2i_weights_*.zip

    Returns:
        Path to the selected zip file, or None if no matching zip is found.
    """
    candidates = sorted(
        repo_root.glob("pixel_t2i_weights_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def safe_extract_zip(zip_path: Path, repo_root: Path) -> None:
    """
    Safely extract a weights zip archive into the repository root.

    This function performs basic security checks before extraction:
    - Ensures the file is a valid zip archive
    - Blocks absolute paths and path traversal entries
    - Verifies that all extracted files are under the expected 'models/' prefix

    Args:
        zip_path: Path to the weights zip archive.
        repo_root: Repository root directory where files will be extracted.

    Raises:
        ValueError: If the zip file is invalid or contains unsafe paths.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    print(f"[info] Zip: {zip_path}")
    print(f"[info] Repo root: {repo_root.resolve()}")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

        for name in names:
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Unsafe entry (absolute path): {name}")
            norm = name.replace("\\", "/")
            if ".." in norm.split("/"):
                raise ValueError(f"Unsafe entry (path traversal): {name}")

        has_prefix = any(
            n.replace("\\", "/").startswith(REQUIRED_PREFIX)
            for n in names
            if not n.endswith("/")
        )
        if not has_prefix:
            raise ValueError(
                f"Zip does not contain required prefix '{REQUIRED_PREFIX}'. "
                "Expected paths like 'models/pixel_unconditional/checkpoints/model_best.pt'."
            )

        z.extractall(repo_root)

    print("[extract] Done.")


def verify_expected_files(repo_root: Path) -> None:
    """
    Verify that key pretrained checkpoint files exist after extraction.

    This is a lightweight sanity check to confirm that the expected model
    weights have been placed under the correct directories.

    Args:
        repo_root: Repository root directory.
    """
    expected = [
        repo_root / "models/pixel_unconditional/checkpoints/model_best.pt",
        repo_root / "models/pixel_text_conditional/checkpoints/model_best.pt",
        repo_root / "models/pixel_image_conditional/checkpoints/unet_best.pt",
        repo_root / "models/pixel_image_conditional/checkpoints/char_encoder_best.pt",
    ]

    missing = [p for p in expected if not p.exists()]
    if missing:
        print("[verify] Warning: some expected files are missing:")
        for p in missing:
            print(f"  - {p}")
        print("[verify] This might be OK if your release uses different filenames.")
    else:
        print("[verify] Looks good: key checkpoint files exist.")


def main() -> int:
    """
    Parse command-line arguments and extract pretrained weights.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Extract PIXEL-T2I pretrained weights zip into repo root."
    )
    parser.add_argument(
        "--zip",
        type=str,
        default="",
        help="Path to weights zip (default: auto-detect in repo root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files (extract anyway).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.zip.strip():
        zip_path = (repo_root / args.zip).resolve() if not Path(args.zip).is_absolute() else Path(args.zip)
    else:
        zip_path = pick_default_zip(repo_root)

    if not zip_path or not zip_path.exists():
        print("[error] Could not find a weights zip.")
        print("Put the zip in the repository root, e.g.: pixel_t2i_weights_v0.1.0.zip")
        print("Or specify it explicitly with: --zip <path>")
        return 2

    if not args.force:
        ckpt_dir = repo_root / "models"
        if ckpt_dir.exists():
            print("[info] Extracting may overwrite existing files. Use --force to proceed anyway.")

    safe_extract_zip(zip_path, repo_root)
    verify_expected_files(repo_root)

    print("[done] Weights extracted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
