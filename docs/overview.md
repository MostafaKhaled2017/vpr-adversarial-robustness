# Overview

## Project purpose

The repository adapts SuperVLAD visual place recognition for adversarial robustness experiments. It includes rank-aware adversarial training, FGSM evaluation, and perceptual attack evaluation against base and trained checkpoints.

## Main technologies

- Python
- PyTorch and torchvision
- NumPy, pandas, scikit-learn, FAISS
- SuperVLAD under `third_party/SuperVLAD`
- `perceptual-advex` under `submodules/perceptual-advex`
- Visual place recognition datasets with database/query image splits

## Main capabilities

- Train SuperVLAD with adversarial rank-aware losses.
- Train with perceptual adversarial attacks from `perceptual-advex`.
- Evaluate clean and FGSM robustness for checkpoints.
- Evaluate base and perceptually trained checkpoints on shared perceptual attack data.
- Produce JSON, CSV, log, checkpoint, and TensorBoard outputs.

## Entry points

- `adv_train.py`: rank-aware adversarial training entry point.
- `perceptual_adv_training.py`: perceptual adversarial training entry point.
- `fgsm_eval.py`: FGSM robustness evaluation entry point.
- `perceptual_eval.py`: clean and perceptual attack comparison entry point.
- `commands.bash`: repository command examples.
- `scripts/run_perceptual_adv_training.sh`: shell wrapper for perceptual adversarial training.

## Notes

The root `README.md` is empty. Most project intent is inferred from Python entry points, command scripts, requirements, and the vendored SuperVLAD README.
