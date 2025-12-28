# Models

This directory contains the **model definitions and inference code** for the
PIXEL-T2I project.

It includes separate modules for unconditional, text-conditioned, and
image-conditioned diffusion models, together with placeholder directories
for pretrained weights and runtime outputs.

This directory does **not** contain training notebooks or experimental logs.

---

## Directory Structure

### `pixel_unconditional/`

Unconditional diffusion model for pixel-art character generation.

- `inference.py`  
  Inference script for generating character sprites without conditioning.

- `checkpoints/`  
  Placeholder directory for pretrained model weights.  
  Kept under version control using `.gitkeep`.

- `temp_outputs/`  
  Temporary directory for inference-time outputs (not tracked).

---

### `pixel_text_conditional/`

Text-conditioned diffusion model for controllable pixel-art character generation.

- `inference.py`  
  Inference script supporting text-guided generation using caption embeddings.

- `checkpoints/`  
  Placeholder directory for pretrained model weights.  
  Kept under version control using `.gitkeep`.

- `temp_outputs/`  
  Temporary directory for inference-time outputs (not tracked).

---

### `pixel_image_conditional/`

Image-conditioned diffusion model for character action generation.

- `image_inference.py`  
  Inference script for generating action frames conditioned on a 4-view character
  image and a frame identifier.

- `checkpoints/`  
  Placeholder directory for pretrained model weights.  
  Kept under version control using `.gitkeep`.

- `temp_outputs/`  
  Temporary directory for inference-time outputs (not tracked).

---

## Notes

- Training is performed in the `experiments/` directory using Jupyter notebooks.
- Pretrained model weights are **not committed directly** to the repository.
  They are distributed separately (e.g. via GitHub Releases) and extracted
  into the corresponding `checkpoints/` directories.
- The `temp_outputs/` directories are used only during inference and are
  intentionally excluded from version control.
