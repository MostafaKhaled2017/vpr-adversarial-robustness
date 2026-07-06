# Dependencies

## System dependencies

- CUDA-capable system is expected for default `--device cuda` runs.
- Exact system packages are not documented in the repository.

## Language dependencies

Pinned in `requirements.txt`:

- `numpy==2.2.6`
- `pandas==2.3.3`
- `Pillow==12.1.1`
- `matplotlib==3.10.9`
- `prettytable==3.17.0`
- `pytorch-lightning==2.6.1`
- `pytorch-metric-learning==2.9.0`
- `scikit-learn==1.7.2`
- `timm==1.0.25`
- `torch==2.11.0`
- `torchvision==0.26.0`
- `tensorboard==2.20.0`
- `tqdm==4.67.3`
- `transformers==4.57.3`
- `faiss-gpu-cu12==1.14.1.post1`

Additional packages in `installation.bash`:

- `advex-uar`
- `recoloradv`
- `PyWavelets`
- `tensorboardX`
- `git+https://github.com/MadryLab/robustness.git`
- `git+https://github.com/fra31/auto-attack.git`
- editable `submodules/perceptual-advex`

## ROS dependencies

Not found in repository.

## ML/CV dependencies

- PyTorch and torchvision
- SuperVLAD
- DINOv2 checkpoint
- perceptual-advex
- FAISS GPU package
- scikit-learn nearest-neighbor utilities

## Dependency notes

The project pins NumPy 2.2.6, where removed aliases such as `np.float` are unavailable. Dataset code has been updated to avoid `np.float`.
