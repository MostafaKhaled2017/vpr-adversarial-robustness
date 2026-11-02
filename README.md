# ARAFT: Anchored Rank-Adversarial Fine-Tuning for Visual Place Recognition

This repository contains the code and scripts behind the paper on ARAFT. It compares ARAFT with plain adversarial
training on SuperVLAD and BoQ. Every result in the paper can be reproduced with the scripts in
`scripts/`.

## Repository layout

| Path                          | Purpose                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `train.py`                    | Training entry point: ARAFT, plain adversarial training and clean fine-tuning.        |
| `eval.py`                     | Test-set evaluation under rank-PGD and rank-APGD attacks.                             |
| `src/`                        | Attacks, losses, training loop, rank evaluation, model adapters (SuperVLAD, BoQ).     |
| `src/paper/`                  | The paper's tables, in-text numbers and figures.                                      |
| `scripts/`                    | Bash scripts that reproduce the paper (below).                                        |
| `tests/`                      | Unit tests; `tests/golden/` holds the arguments of every run the paper reports.       |
| `third_party/`                | SuperVLAD, Bag-of-Queries and the VPR benchmark's dataset code.                       |
| `submodules/perceptual-advex` | Perceptual attacks, used only by the attack-set screens of the hyperparameter search. |

## Installation

Python 3.12 on Linux x86-64 is the reference environment.

```bash
git submodule update --init
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The attack-set screens (`scripts/train_screens.sh`) also need the perceptual attack stack:

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

## Reproducing the paper

Run the scripts in this order from the repository root. Each one can be stopped and rerun: finished
runs and evaluations are skipped, and a stopped training run resumes from its last checkpoint.
Training runs go to `logs/paper/<run>/` and evaluations to `output/paper/<model>/<attack>/<dataset>/<tag>/`.

| Step | Script                       | What it produces                                                  | Paper                     |
| ---- | ---------------------------- | ----------------------------------------------------------------- | ------------------------- |
| 1    | `scripts/train_screens.sh`   | 13 SuperVLAD screens of 12 epochs (hyperparameter search)         | Table 5                   |
| 2    | `scripts/train_supervlad.sh` | ARAFT (two seeds), plain AT, the clean twin and four ablations    | Tables 2–4, 6; Fig. 1–3   |
| 3    | `scripts/train_boq.sh`       | ARAFT and plain AT on BoQ, with the SuperVLAD settings            | Tables 2, 3, 6; Fig. 1, 3 |
| 4    | `scripts/eval_supervlad.sh`  | Rank-APGD / rank-PGD test results and the gradient-masking checks | Tables 2–4, 6; Sect. 5    |
| 5    | `scripts/eval_boq.sh`        | Rank-APGD / rank-PGD test results on BoQ                          | Tables 2, 3, 6            |

The full run takes several hundred GPU-hours. A full-length training run takes one to two and a half
days, and a 100-step rank-APGD evaluation of one model on MSLS takes about 2.5 hours. Training at the
paper's batch of 20 places × 4 images needs a GPU with more than 10 GB of memory.

`scripts/train.sh` and `scripts/eval.sh` run a single training run or evaluation; see their headers
for the options.

## Tests

```bash
python -m unittest discover tests
```

`tests/test_paper_scripts.py` checks that the scripts run exactly the configurations recorded for
the paper. `tests/test_paper_tables.py` checks the paper's printed numbers; it needs the results in
`logs/paper` and `output/paper`.
