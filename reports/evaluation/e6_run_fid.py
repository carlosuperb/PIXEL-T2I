"""
e6_run_fid.py

This script computes Fréchet Inception Distance (FID) scores between real and
generated images for different models.

It evaluates three settings: unconditional generation, text-conditioned
generation, and image-conditioned action generation, providing a consistent
comparison across tasks.
"""

import subprocess
from pathlib import Path


# base evaluation directory
BASE_DIR = Path("reports/evaluation")

# real datasets (RGB-converted)
REAL_4VIEW = BASE_DIR / "real_4view_rgb"
REAL_ACTION = BASE_DIR / "real_action_rgb"

# generated datasets (RGB-converted)
FAKE_UNCOND = BASE_DIR / "unconditional/fake_uncond_rgb"
FAKE_TEXT = BASE_DIR / "text_conditional/fake_text_rgb"
FAKE_ACTION = BASE_DIR / "image_conditional/fake_action_rgb"


def run_fid(real_dir, fake_dir, name):
    """
    Run FID computation between a pair of real and generated image directories.

    Args:
        real_dir (Path): Directory containing real images.
        fake_dir (Path): Directory containing generated images.
        name (str): Descriptive name of the evaluation setting.

    Returns:
        str: "Done" if execution succeeds, otherwise "Error".

    Notes:
        - Uses pytorch_fid for evaluation.
        - num-workers is set to 0 to avoid multiprocessing issues on some systems.
        - Results are printed directly to the terminal by pytorch_fid.
    """

    # print task information
    print(f"\nRunning FID for: {name}")
    print(f"Real: {real_dir}")
    print(f"Fake: {fake_dir}")

    # check if directories exist
    if not real_dir.exists():
        raise FileNotFoundError(f"Real directory not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake directory not found: {fake_dir}")

    # build command (quoted paths for cross-platform safety)
    cmd = (
        f'python -m pytorch_fid "{real_dir}" "{fake_dir}" '
        f'--device cuda --batch-size 32 --num-workers 0'
    )

    # print command for debugging
    print(f"Command: {cmd}")

    # execute command (output printed directly)
    result = subprocess.run(cmd, shell=True)

    # return execution status
    if result.returncode == 0:
        return "Done"
    else:
        return "Error"


def main():
    """
    Run FID evaluation across all model settings and print a summary.

    Evaluations:
        - Unconditional model on 4-view dataset
        - Text-conditioned model on 4-view dataset
        - Image-conditioned model on action dataset
    """

    results = []

    # unconditional (4-view)
    fid_uncond = run_fid(REAL_4VIEW, FAKE_UNCOND, "Unconditional (4-view)")
    results.append(("Unconditional", "4-view", fid_uncond))

    # text-conditioned (4-view)
    fid_text = run_fid(REAL_4VIEW, FAKE_TEXT, "Text-conditioned (4-view)")
    results.append(("Text-conditioned", "4-view", fid_text))

    # image-conditioned (action)
    fid_action = run_fid(REAL_ACTION, FAKE_ACTION, "Image-conditioned (Action)")
    results.append(("Image-conditioned", "Action", fid_action))

    # print summary
    print("\n===== FID Summary =====")
    for model, dataset, fid in results:
        print(f"{model} | {dataset} | {fid}")


if __name__ == "__main__":
    main()