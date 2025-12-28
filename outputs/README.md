# Outputs (Training Artefacts)

This directory contains **training artefacts** produced during the training of diffusion models in this project.

It is **not** used for runtime inference or web demo outputs.

## Contents

The outputs are organised by conditioning type:

- `logs_unconditional/`  
  Epoch-level training loss logs (CSV format) for the unconditional diffusion model.

- `logs_text_conditional/`  
  Epoch-level training loss logs (CSV format) for the text-conditional diffusion model.

- `logs_image_conditional/`  
  Epoch-level training loss logs (CSV format) for the image-conditional diffusion model.

- `samples_unconditional/`  
  Representative samples generated at selected training epochs for qualitative evaluation.

- `samples_text_conditional/`  
  Representative samples generated at selected training epochs for qualitative evaluation.

- `samples_image_conditional/`  
  Representative samples generated at selected training epochs for qualitative evaluation.

## Notes

- The logs are used to generate loss curves reported in the project report.
- The sample images are included for qualitative analysis of training behaviour and convergence.
- Runtime inference outputs (e.g. web demo results) are stored separately and are **not** tracked under this directory.
