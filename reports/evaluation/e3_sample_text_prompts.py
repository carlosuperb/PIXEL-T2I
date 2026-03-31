"""
e3_sample_text_prompts.py

This script randomly samples a subset of text prompts from the caption dataset
and saves them to a text file for evaluation.

The sampled prompts are used as inputs for text-conditioned generation, ensuring
a consistent and reproducible set of prompts when evaluating semantic alignment
(e.g., CLIP score or qualitative comparison).
"""

import csv
import random
from pathlib import Path

# caption file (processed dataset with optimized text descriptions)
CAPTION_FILE = Path("pixel_character_dataset/processed/dataset_4view/captions_optimized.csv")

# output file for sampled prompts
OUTPUT_FILE = Path("reports/evaluation/text_conditional/prompts_1000.txt")

# sampling configuration
NUM_SAMPLES = 1000
SEED = 42


def main():
    """
    Sample a fixed number of text prompts from the caption dataset and save them
    to a text file.

    The sampling is performed without replacement and controlled by a fixed
    random seed to ensure reproducibility across evaluation runs.

    Raises:
        ValueError: if the dataset contains fewer captions than NUM_SAMPLES.
    """

    # ensure deterministic sampling
    random.seed(SEED)

    # read all captions from CSV file
    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        captions = [row["text"] for row in reader]

    print(f"Total captions: {len(captions)}")

    # check dataset size is sufficient
    if len(captions) < NUM_SAMPLES:
        raise ValueError("Not enough captions")

    # randomly select prompts (no replacement)
    selected = random.sample(captions, NUM_SAMPLES)

    # create output directory if it does not exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # save prompts to file (one prompt per line)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prompt in selected:
            f.write(prompt.strip() + "\n")

    print(f"Saved {NUM_SAMPLES} prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()