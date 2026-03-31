# Reports

This directory contains all resources related to **evaluation and reporting**
for the PIXEL-T2I project.

It includes evaluation scripts, prepared datasets for metric computation,
and figures used in the final report.

The contents are primarily used for **offline evaluation and analysis**,
and are not part of the model training or inference pipeline.

---

## Directory Structure

### `evaluation/`

This folder contains datasets and scripts used for quantitative evaluation,
including FID and CLIP score computation.

#### Evaluation Scripts (`e*`)

Scripts prefixed with `e` implement the full evaluation pipeline.

- **`e1_sample_real_4view_images.py`**  
  Sample a subset of real 4-view character images for evaluation.

- **`e2_sample_real_action_images.py`**  
  Sample a subset of real action sprite sheets for evaluation.

- **`e3_sample_text_prompts.py`**  
  Sample text prompts from the caption dataset for text-conditioned evaluation.

- **`e4_convert_to_rgb.py`**  
  Convert RGBA images to RGB format for compatibility with FID.

- **`e5_convert_to_rgb_threshold.py`**  
  Improved RGBA to RGB conversion using alpha thresholding to reduce background artefacts.

- **`e6_run_fid.py`**  
  Compute FID scores for unconditional, text-conditioned, and image-conditioned models.

- **`e7_run_clip.py`**  
  Compute CLIP scores to evaluate semantic alignment between images and prompts.

---

#### Evaluation Datasets

These directories contain **real and generated images** used for evaluation.

- **`real_4view/`**  
  Sampled real 4-view character images.

- **`real_4view_rgb/`**  
  RGB-converted version of `real_4view/` for FID computation.

- **`real_action/`**  
  Sampled real action sprite sheets.

- **`real_action_rgb/`**  
  RGB-converted version of `real_action/`.

---

#### Generated Outputs

Each subdirectory corresponds to a model type.

- **`unconditional/`**
  - `fake_uncond/`: Generated 4-view images  
  - `fake_uncond_rgb/`: RGB-converted images  

- **`text_conditional/`**
  - `fake_text/`: Generated images from text prompts  
  - `fake_text_rgb/`: RGB-converted images  
  - `prompts_1000.txt`: Sampled prompts used for evaluation  

- **`image_conditional/`**
  - `fake_action/`: Generated action sprite sheets  
  - `fake_action_rgb/`: RGB-converted images  

---

### `figures/`

This directory contains all figures used throughout the report, including:

- system architecture diagrams  
- dataset construction visualisations  
- model design illustrations  
- qualitative generation results  
- evaluation plots

These figures are referenced across different sections of the report (e.g., Introduction, Background, Methodology, and Evaluation) and are not used during training or inference.

---

## Evaluation Pipeline

The evaluation process follows a structured pipeline:

1. Sample real datasets (`e1`, `e2`)  
2. Sample text prompts (`e3`)  
3. Generate images using model inference scripts (`models/`)  
4. Convert images to RGB (`e4`, `e5`)  
5. Compute metrics:  
   - FID (`e6`)  
   - CLIP score (`e7`)  

This ensures a **consistent and reproducible evaluation setup** across all models.

---

## Reproducibility and Data Availability

Due to repository size constraints, evaluation images (both real and generated)
are not included.

All results can be reproduced using the provided scripts together with the
model inference code:

- Real datasets are generated via:
  - `e1_sample_real_4view_images.py`
  - `e2_sample_real_action_images.py`

- Text prompts are generated via:
  - `e3_sample_text_prompts.py`

- Generated images should be produced using the inference scripts in `models/`
  and saved to the following locations:

  - `evaluation/unconditional/fake_uncond/`
  - `evaluation/text_conditional/fake_text/`
  - `evaluation/image_conditional/fake_action/`

- RGB conversion:
  - `e4_convert_to_rgb.py` or `e5_convert_to_rgb_threshold.py`

- Metric computation:
  - `e6_run_fid.py`
  - `e7_run_clip.py`

All sampling steps use fixed random seeds to ensure reproducibility.

---

## Notes

- Evaluation is performed on **fixed sampled subsets** to ensure fairness and reproducibility.  
- All images are converted to RGB before FID computation, as required by the Inception model.  
- The threshold-based conversion (`e5`) is used to reduce artefacts from transparent backgrounds.  
- Generated outputs are organised by model type to support direct comparison.