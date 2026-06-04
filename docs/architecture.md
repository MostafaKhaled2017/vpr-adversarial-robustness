# Architecture

## High-level architecture

The project layers adversarial training and evaluation code on top of SuperVLAD. SuperVLAD supplies datasets, model construction, parser defaults, logging, and evaluation utilities. The local `perceptual_adv_training` package adds retrieval-specific attack targets, attack wrappers, losses, checkpoint handling, and training loops.

## Main modules

- `adv_train.py`: adversarial fine-tuning script using SuperVLAD data/model utilities and rank-aware losses.
- `fgsm_eval.py`: differentiable FGSM attack evaluation for visual geolocalization checkpoints.
- `perceptual_eval.py`: compares base and trained checkpoints on clean and perceptual attack results.
- `perceptual_adv_training/`: package for perceptual adversarial training components.
- `third_party/SuperVLAD/`: SuperVLAD implementation, model code, parser, datasets, training, and evaluation utilities.
- `submodules/perceptual-advex/`: perceptual adversarial attack implementations used by local wrappers.
- `scripts/`: shell wrappers for common training and evaluation runs.

## Data flow

Evaluation datasets are expected under `--eval_datasets_folder` with `images/<split>/database` and `images/<split>/queries` folders. SuperVLAD dataset code loads image paths and UTM coordinates from filenames, extracts database and query descriptors, then computes recalls. Attack scripts generate adversarial query tensors, evaluate descriptor retrieval behavior, and write logs plus structured result files.

Training uses GSV-Cities data through `perceptual_adv_training/data.py`, builds SuperVLAD training components, generates adversarial queries, computes retrieval losses, validates on configured evaluation datasets, and writes checkpoints/logs.

## External dependencies

- DINOv2 foundation checkpoint, commonly `checkpoints/dinov2_vitb14_pretrain.pth`.
- SuperVLAD checkpoint, commonly `checkpoints/SuperVLAD.pth`.
- Evaluation datasets such as `msls`, `sped`, and `nordland` as shown in `commands.bash`.
- GSV-Cities training dataset.
- CUDA-capable PyTorch environment for GPU runs when `--device cuda` is used.

## Design notes

The repository depends on `PYTHONPATH` including the repository root and `third_party/SuperVLAD`, as shown in `commands.bash`. Dataset UTM parsing relies on image filenames containing `@utm_easting@utm_northing@...@.jpg`.
