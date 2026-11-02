# ARAFT: Anchored Rank-Adversarial Fine-Tuning for Visual Place Recognition

This repository contains the code for ARAFT, with scripts in `scripts/` that train and evaluate
ARAFT on SuperVLAD and BoQ.

## Repository layout

| Path                          | Purpose                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `train.py`                    | Training entry point: ARAFT, plain adversarial training and clean fine-tuning.    |
| `eval.py`                     | Test-set evaluation under rank-PGD and rank-APGD attacks.                         |
| `src/`                        | Attacks, losses, training loop, rank evaluation, model adapters (SuperVLAD, BoQ). |
| `scripts/`                    | ARAFT training and evaluation scripts (below).                                    |
| `tests/`                      | Unit tests.                                                                       |
| `third_party/`                | SuperVLAD, Bag-of-Queries and the VPR benchmark's dataset code.                   |
| `submodules/perceptual-advex` | Attack backends imported by `src/attacks.py` during training.                     |

## Installation

Python 3.12 on Linux x86-64 is the reference environment.

```bash
git submodule update --init
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training also needs the attack backends of `submodules/perceptual-advex`:

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

## Training and evaluation examples

Run the scripts from the repository root. Each one can be stopped and rerun: finished runs and
evaluations are skipped, and a stopped training run resumes from its last checkpoint. Training runs
go to `logs/<run>/` and evaluations to `output/<model>/<attack>/<dataset>/<tag>/`.

| Script                       | What it does                                                            |
| ---------------------------- | ----------------------------------------------------------------------- |
| `scripts/train_supervlad.sh` | Trains ARAFT on SuperVLAD into `logs/supervlad_araft_s0/`.              |
| `scripts/train_boq.sh`       | Trains ARAFT on BoQ into `logs/boq_araft_s0/`.                          |
| `scripts/eval_supervlad.sh`  | Rank-APGD and rank-PGD test results of the SuperVLAD ARAFT checkpoint.  |
| `scripts/eval_boq.sh`        | Rank-APGD and rank-PGD test results of the BoQ ARAFT checkpoint.        |

The evaluation scripts cover MSLS, Pitts30k, SPED and Nordland, with rank-APGD at 2/255 and 4/255
and rank-PGD at 1, 2, 4 and 8/255.

```bash
# Train ARAFT on SuperVLAD, then evaluate it
scripts/train_supervlad.sh
scripts/eval_supervlad.sh

# Evaluate a downloaded ARAFT checkpoint
CHECKPOINT=checkpoints/<araft_boq>.pth scripts/eval_boq.sh

# One evaluation: SuperVLAD on MSLS under rank-APGD at 4/255
scripts/eval.sh supervlad apgd msls araft logs/supervlad_araft_s0/best_model.pth 0.0685

# ARAFT on SuperVLAD with the closest positive and anchor weight 2
scripts/train.sh supervlad_araft_closest_aw2_s0 supervlad araft 0 POSITIVE=closest AW=2
```

A full-length training run takes one to two and a half days, and a 100-step rank-APGD evaluation of
one model on MSLS takes about 2.5 hours. Training at a batch of 20 places × 4 images needs a GPU with
more than 10 GB of memory.

`scripts/train.sh` and `scripts/eval.sh` run a single training run or evaluation; see their headers
for the options.

## Tests

```bash
python -m unittest discover tests
```
