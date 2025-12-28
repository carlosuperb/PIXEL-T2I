# Experiments

This directory contains the **training notebooks used for all core diffusion models**
in the PIXEL-T2I project.

The notebooks in this folder represent the **experimental backbone** of the project
and are used to train, evaluate, and compare different conditioning strategies
under a unified diffusion framework.

These notebooks are **not part of the runtime inference or web demo pipeline**.

---

## Overview

The experiments are organised as a **progressive sequence of models**:

1. **Unconditional diffusion** (baseline)
2. **Text-conditioned diffusion**
3. **Image-conditioned diffusion**

Each notebook builds conceptually on the previous one, enabling controlled
comparisons across conditioning types while keeping architectural and training
settings as consistent as possible.

---

## Notebooks

- **`n1_train_unconditional_diffusion.ipynb`**  
  Trains an unconditional diffusion model on 50,000 RGBA pixel-art character sprites.  
  Serves as the baseline for evaluating diversity, convergence behaviour, and
  sample quality without conditioning.

- **`n2_train_text_conditional_diffusion.ipynb`**  
  Trains a text-conditioned diffusion model using optimised natural-language captions.  
  Extends the unconditional baseline by introducing textual conditioning, enabling
  controllable sprite generation and direct qualitative and quantitative comparison
  with `n1`.

- **`n3_train_image_conditional_diffusion.ipynb`**  
  Trains an image-conditioned diffusion model for character action generation.  
  The model predicts individual action tiles conditioned on a 4-view character image
  and an explicit frame identifier, which are later assembled into a full LPC-style
  action sprite sheet during inference.

---

## Outputs and Checkpoints

- **Training logs and qualitative samples** generated during training are stored under:

  ```text
  outputs/
  ```

  These include epoch-level loss logs (CSV) and representative generated samples
  used for analysis and reporting.

- **Model checkpoints** are saved externally (e.g. Google Drive) during training  
  and are later imported into the repository for inference or release packaging.

---

## Notes

- These notebooks are used to **train and evaluate models**, not to perform
  interactive image generation or serve as a deployed system.
- GPU resources are required for full training runs; the reported experiments
  were conducted using high-performance GPUs (NVIDIA A100), but other modern GPUs
  may also be suitable with adjusted batch sizes or training settings.
- Detailed setup instructions, assumptions, and execution details are provided
  at the top of each notebook.

