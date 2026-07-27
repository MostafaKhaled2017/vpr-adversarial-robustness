# VPR Adversarial Robustness

Adversarial robustness experiments for Visual Place Recognition (VPR). The repository adapts VPR models such as SuperVLAD, BoQ and MixVPR for adversarial training and evaluation, including retrieval-native rank attacks, FGSM evaluation, and perceptual (LPIPS-bounded) attack evaluation against base and adversarially trained checkpoints.

## Capabilities

- Train VPR models with rank-aware adversarial losses or with perceptual adversarial attacks from `perceptual-advex`.
- Evaluate checkpoints under retrieval-native rank attacks (Rank-PGD L∞/L2, APGD-style) with white-box per-model or shared transfer protocols.
- Evaluate clean vs. FGSM robustness and clean vs. perceptual-attack robustness.
- Run budgeted Rank-PGD strength sweeps with resume support.
- Produce JSON, CSV, log, checkpoint, and TensorBoard outputs, plus attack-image artifacts for report figures.

## Repository layout

| Path                          | Purpose                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `eval.py`                     | Unified evaluation entry point. Rank attacks by default; `eval.py perceptual` for the clean/perceptual comparison. |
| `train.py`                    | Perceptual adversarial training entry point.                                                                       |
| `src/`                        | Local package: rank/perceptual/FGSM evaluators, attacks, losses, targets, metrics, model registry.                 |
| `src/rank_eval.py`            | Native rank attack evaluation implementation (also directly runnable).                                             |
| `src/perceptual_eval.py`      | Clean and perceptual attack comparison implementation (also directly runnable).                                    |
| `src/fgsm_eval.py`            | FGSM robustness evaluation script (directly runnable).                                                             |
| `src/fgsm_train.py`           | Rank-aware adversarial training script (`python -m src.fgsm_train`).                                               |
| `src/models/`                 | Extensible model registry and SuperVLAD/BoQ/MixVPR adapters.                                                       |
| `scripts/`                    | Shell wrappers for training/evaluation.                                                                            |
| `third_party/`                | Vendored SuperVLAD and related VPR code.                                                                           |
| `submodules/perceptual-advex` | Perceptual attack library (LPIPS-bounded attacks).                                                                 |
| `tests/`                      | Unit tests (`python -m unittest discover tests`).                                                                  |
| `checkpoints/`                | Local model weights (never downloaded automatically).                                                              |
| `datasets/`                   | Evaluation datasets (see layout below).                                                                            |

## Installation

Python 3.12 on Linux x86-64 is the reference environment.

```bash
python -m venv venv
pip install -r requirements.txt
```

Additional dependencies used by the perceptual attack stack:

```bash
python -m pip install advex-uar recoloradv PyWavelets tensorboardX
python -m pip install git+https://github.com/MadryLab/robustness.git
python -m pip install git+https://github.com/fra31/auto-attack.git
```

## Data and checkpoints

Evaluation datasets live under `datasets/<name>/images/test/` with `database/` and `queries/` splits (e.g. `msls`, `sped`, `nordland`). Checkpoints for the adversarially trained models are available in the [Google Drive checkpoint folder](https://drive.google.com/drive/folders/1dP61euhUI2I5e9e-FE1A_Vvf09b-fLQ1?usp=sharing).

Download the required checkpoints to `checkpoints/` before running an evaluation; the evaluation scripts validate paths up front and never download weights automatically. SuperVLAD additionally needs the DINOv2 foundation weights (`checkpoints/dinov2_vitb14_pretrain.pth`).

## Usage

The commands below show the shape of each workflow with `<placeholders>`. Ready-to-run examples with real flag values are in the `scripts/` directory.

### Training

```bash
python train.py \
  --eval_datasets_folder=<datasets-folder> --eval_dataset_name=<dataset> \
  --attack "<perceptual-attack-expression>" \
  <model-flags>
```

Rank-aware adversarial training:

```bash
python -m src.fgsm_train \
  --eval_datasets_folder=<datasets-folder> --eval_dataset_name=<dataset> \
  --resume=<checkpoint.pth> \
  --adv_epsilon=<epsilon> \
  <model-flags>
```

### Rank attack evaluation

```bash
python eval.py \
  --model_type=<supervlad|boq|mixvpr> \
  --model_paths <checkpoint.pth> [<another-checkpoint.pth> ...] \
  --eval_datasets_folder=<datasets-folder> --datasets <dataset> [...] \
  --epsilons <epsilon> [...] \
  <model-flags>
```

Pass multiple `--model_paths` (with `--model_tags`) to compare checkpoints; add `--shared_attacks` for the transfer protocol instead of per-model white-box attacks. See `python eval.py --help` for the full interface.

### Perceptual attack comparison

```bash
python eval.py perceptual \
  --datasets <dataset> [...] \
  --base_resume <base-checkpoint.pth> \
  --trained_resume <trained-checkpoint.pth>
```

### FGSM evaluation

```bash
python src/fgsm_eval.py \
  --eval_datasets_folder=<datasets-folder> --eval_dataset_name=<dataset> \
  --resume=<checkpoint.pth> \
  --epsilons <epsilon> [...] --fgsm_loss <loss> \
  <model-flags>
```

## Tests

```bash
python -m unittest discover tests
```

## Outputs

Evaluation runs write timestamped JSON/CSV reports and logs under `test/rank_eval/` and `test/perceptual_eval/`; training runs write logs, checkpoints, and TensorBoard files under `logs/`.
