# Setup

## Environment

Expected OS and Python version are not explicitly documented. The active environment that reported the NumPy issue used Python 3.12. `requirements.txt` pins CUDA 12 PyTorch packages from the PyTorch extra index.

## Dependencies

Repository-documented installation snippets:

- `python -m pip install advex-uar recoloradv PyWavelets tensorboardX` from `installation.bash`.
- `python -m pip install git+https://github.com/MadryLab/robustness.git` from `installation.bash`.
- `python -m pip install git+https://github.com/fra31/auto-attack.git` from `installation.bash`.
- `python -m pip install -e submodules/perceptual-advex --no-deps` from `installation.bash`.

Inferred standard setup from `requirements.txt`:

- `python3 -m pip install -r requirements.txt`

## Configuration

- `PYTHONPATH="$PWD:$PWD/third_party/SuperVLAD:${PYTHONPATH:-}"` is shown in `commands.bash`.
- `--eval_datasets_folder` points to evaluation datasets.
- `--gsv_cities_base_path` or `GSV_CITIES_BASE_PATH` points to GSV-Cities training data.
- `--foundation_model_path` points to the DINOv2 checkpoint.
- `--resume`, `--base_resume`, and `--trained_resume` point to SuperVLAD or trained checkpoints.

## Unknowns

- Full OS support: Unknown.
- Whether all pinned package versions install together in a fresh environment: Not validated.
- Exact checkpoint download locations for this repository: Not found in repository, except SuperVLAD README links.
