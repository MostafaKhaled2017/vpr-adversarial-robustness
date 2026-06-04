# Inference

## Descriptor extraction

SuperVLAD model inference extracts descriptors for database and query images. The local training package contains helper functions in `perceptual_adv_training/data.py` for extracting database and clean query features.

## Entry points

- `third_party/SuperVLAD/eval.py`: upstream SuperVLAD evaluation entry point.
- `fgsm_eval.py`: loads a checkpoint and evaluates clean plus FGSM-perturbed query descriptors.
- `perceptual_eval.py`: loads base and trained checkpoints for clean and perceptual attack comparisons.

## Notes

Inference requires dataset folders and checkpoint paths. Full inference was not run during initial documentation.
