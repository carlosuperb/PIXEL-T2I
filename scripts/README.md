# Scripts

This directory contains utility scripts used for **dataset construction, preprocessing,
training analysis, and release preparation** in the PIXEL-T2I project.

The scripts are **not part of the runtime inference pipeline**.
They are intended for offline data generation, inspection, evaluation, and packaging.

## Overview by Category

The scripts are grouped by prefix to reflect their purpose and typical usage stage.

---

### Dataset Construction & Preprocessing (`t*`)

Scripts prefixed with `t` are primarily used to **build and inspect training datasets**
from LPC-style layered assets.

- **`t1_assemble_full_sheet_demo.py`**  
  Assemble a full LPC-style layered character sprite sheet.  
  Used mainly for debugging layer order, alignment, and sheet consistency.

- **`t2_build_4view_from_sheet_demo.py`**  
  Extract four static standing poses (north, west, south, east) from a full sheet
  and arrange them into a 2×2 four-view grid.

- **`t3_generate_dataset_4view.py`**  
  Generate a large dataset of 2×2 four-view character sprites together with
  natural-language captions for diffusion / LoRA training.

- **`t4_lpc_caption_utils.py`**  
  Utility functions for converting LPC asset paths into descriptive caption phrases.
  Intended to be imported by other scripts.

- **`t5_optimize_captions.py`**  
  Post-process and rewrite captions into a shorter, more structured, LoRA-friendly format
  while preserving all visual attributes.

- **`t6_preview_dataset.py`**  
  Qualitatively inspect the dataset by sampling images, comparing original vs optimized
  captions, and generating preview figures and tables.

- **`t7_export_captions_txt.py`**  
  Export per-image caption `.txt` files (both original and optimized) alongside the dataset,
  without copying image files.

---

### Training Analysis (`t8_*`)

- **`t8_plot_loss.py`**  
  Plot training loss curves from CSV logs and save publication-ready figures.
  Used for report and dissertation visualisation.

---

### Action Sheet Generation (`u*`)

Scripts prefixed with `u` are used to **build LPC-style action sprite sheets**
(walk / thrust / slash) from assembled assets.

- **`u1_build_actions_from_sheet_demo.py`**  
  Extract and stack multiple action blocks from a single assembled full sheet
  into one combined action sprite sheet.

- **`u2_generate_dataset_actions.py`**  
  Generate a dataset of combined action sheets (walk + thrust + slash) by sampling
  layered assets, without generating captions.

---

### Model Release Utilities (`v*`)

Scripts prefixed with `v` are used for **packaging and distributing pretrained models**.

- **`v0_build_release_weights.py`**  
  Build a clean, release-ready ZIP archive containing pretrained model weights,
  suitable for GitHub Releases.

- **`v1_extract_weights_zip.py`**  
  Extract pretrained model weights from a release ZIP into the correct repository
  structure.

---

## Notes

- These scripts are designed to be run manually from the repository root.
- Most scripts are **one-off or stage-specific utilities**, not part of the core
  training or inference code.
- Refer to individual script headers for detailed usage and assumptions.
