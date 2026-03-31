"""
e7_run_clip.py

This script computes the CLIP score to evaluate semantic alignment between
generated images and their corresponding text prompts.

Each image is paired with a prompt in a one-to-one manner, and similarity is
computed in a shared embedding space using a pretrained CLIP model.
"""

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm


def compute_clip_score(image_dir, prompt_file, device="cuda"):
    """
    Compute the average CLIP score for a set of generated images and prompts.

    Args:
        image_dir (str): Directory containing generated images.
        prompt_file (str): Text file with one prompt per line (ordered).
        device (str): Device for computation ("cuda" or "cpu").

    Returns:
        float: Average cosine similarity between image and text embeddings.

    Notes:
        - Images and prompts are matched by index (generated_0000.png ↔ line 0).
        - Images are converted to RGB for compatibility with CLIP.
        - Cosine similarity is computed after feature normalization.
    """

    # load pretrained CLIP model and preprocessing pipeline
    model, preprocess = clip.load("ViT-B/32", device=device)

    # read prompts (one-to-one correspondence with generated images)
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f]

    scores = []

    # iterate through prompts and match corresponding images
    for i, text in enumerate(tqdm(prompts)):
        fname = f"generated_{i:04d}.png"
        image_path = os.path.join(image_dir, fname)

        # skip missing images
        if not os.path.exists(image_path):
            continue

        # load and preprocess image (RGB required for CLIP)
        image = preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0).to(device)

        # tokenize text prompt
        text_input = clip.tokenize([text]).to(device)

        with torch.no_grad():
            # extract embeddings
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_input)

            # normalize to unit vectors (required for cosine similarity)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # compute cosine similarity
            similarity = (image_features @ text_features.T).item()

            scores.append(similarity)

    # return average CLIP score
    return sum(scores) / len(scores) if scores else 0


if __name__ == "__main__":
    score = compute_clip_score(
        image_dir="reports/evaluation/text_conditional/fake_text_rgb",
        prompt_file="reports/evaluation/text_conditional/prompts_1000.txt",
    )

    print(f"CLIP Score: {score:.4f}")