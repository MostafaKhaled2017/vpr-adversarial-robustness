# ARAFT: Anchored Rank-Adversarial Fine-Tuning for Visual Place Recognition

This repository contains the code and scripts behind the paper on ARAFT. It compares ARAFT with plain adversarial
training on SuperVLAD and BoQ. Every result in the paper can be reproduced with the scripts in
`scripts/`.

## Repository layout

| Path                          | Purpose                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `train.py`                    | Training entry point: ARAFT, plain adversarial training and clean fine-tuning.    |
| `eval.py`                     | Test-set evaluation under rank-PGD and rank-APGD attacks.                         |
| `src/`                        | Attacks, losses, training loop, rank evaluation, model adapters (SuperVLAD, BoQ). |
| `scripts/`                    | Bash scripts that reproduce the paper (below).                                    |
| `tests/`                      | Unit tests.                                                                       |
| `third_party/`                | SuperVLAD, Bag-of-Queries and the VPR benchmark's dataset code.                   |
| `submodules/perceptual-advex` | Perceptual attacks, used only by `scripts/train.sh MIX=all`.                      |

## Installation

Python 3.12 on Linux x86-64 is the reference environment.

```bash
git submodule update --init
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training with `MIX=all` in `scripts/train.sh` also needs the perceptual attack stack:

```bash
python -m pip install advex-uar recoloradv PyWavelets tensorboardX
python -m pip install git+https://github.com/MadryLab/robustness.git
```

## Data and checkpoints

- **Datasets.** `datasets/gsv_cities` is the training data (GSV-Cities). MSLS-val is used for
  validation. The test sets are `datasets/{msls,pitts30k,sped,nordland}/images/test/`, each with
  `database/` and `queries/` splits, in the format of the VPR benchmark of Berton et al.
- **Checkpoints.** Put these in `checkpoints/`; nothing is downloaded automatically:
  - `SuperVLAD_base.pth`: the SuperVLAD release trained on GSV-Cities.
  - `dinov2_vitb14_pretrain.pth`: the DINOv2 ViT-B/14 weights SuperVLAD builds on.
  - `boq_dinov2_12288.pth`: the BoQ DINOv2 release.

## Trained models

The ARAFT checkpoints of SuperVLAD and BoQ are available on
[Google Drive](https://drive.google.com/drive/folders/1dP61euhUI2I5e9e-FE1A_Vvf09b-fLQ1?usp=sharing).

## Reproducing the paper

Run the scripts in this order from the repository root. Each one can be stopped and rerun: finished
runs and evaluations are skipped, and a stopped training run resumes from its last checkpoint.
Training runs go to `logs/<run>/` and evaluations to `output/<model>/<attack>/<dataset>/<tag>/`.

| Step | Script                       | What it produces                                                  | Paper                     |
| ---- | ---------------------------- | ----------------------------------------------------------------- | ------------------------- |
| 1    | `scripts/train_supervlad.sh` | ARAFT (two seeds), plain AT, the clean twin and four ablations    | Tables 2–4, 6; Fig. 1–3   |
| 2    | `scripts/train_boq.sh`       | ARAFT and plain AT on BoQ, with the SuperVLAD settings            | Tables 2, 3, 6; Fig. 1, 3 |
| 3    | `scripts/eval_supervlad.sh`  | Rank-APGD / rank-PGD test results and the gradient-masking checks | Tables 2–4, 6; Sect. 5    |
| 4    | `scripts/eval_boq.sh`        | Rank-APGD / rank-PGD test results on BoQ                          | Tables 2, 3, 6            |

The full run takes several hundred GPU-hours. A full-length training run takes one to two and a half
days, and a 100-step rank-APGD evaluation of one model on MSLS takes about 2.5 hours. Training at the
paper's batch of 20 places × 4 images needs a GPU with more than 10 GB of memory.

`scripts/train.sh` and `scripts/eval.sh` run a single training run or evaluation; see their headers
for the options.

## Tests

```bash
python -m unittest discover tests
```
