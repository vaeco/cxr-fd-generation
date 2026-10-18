# Method and repository notice

This repository implements a conditional GAN experiment with representation-space
Fréchet loss. It does not implement the complete PIXEL training and localization
system.

- Differentiable Fréchet loss and stop-gradient normalization are based on
  Representation Fréchet Loss for Visual Generation and the FD-Loss repository:
  https://github.com/Jiawei-Yang/FD-Loss
- The chest X-ray conditional generation setting is related to PIXEL:
  https://doi.org/10.1038/s41551-025-01456-y
- Neither upstream archive is bundled or required at runtime. Refer to the
  original papers and repositories when using or extending these methods.
- CheXpert images, patient-level manifests, and model checkpoints are not
  distributed here. Data access and upstream model terms apply independently
  of the MIT license for this implementation.
