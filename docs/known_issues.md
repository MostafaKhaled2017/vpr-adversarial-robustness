# Known Issues

## `rank_eval.py` used `test/rank_eval` itself as the run directory

**Symptom**

`python3 rank_eval.py ... --output_json=test/rank_eval/... --output_csv=test/rank_eval/...` failed with `FileExistsError: test/rank_eval already exists!`.

**Context**

The failure happened before model loading when `rank_eval.py` called SuperVLAD `commons.setup_logging(args.save_dir)`.

**Likely cause**

When explicit output paths were provided under `test/rank_eval`, `rank_eval.py` used the output file parent directory as `args.save_dir`. SuperVLAD logging refuses to use an existing output directory.

**Fix or workaround**

`rank_eval.py` now creates a timestamped run directory under the requested output parent and writes the requested JSON/CSV filenames inside that directory. For example, `--output_json=test/rank_eval/results.json` writes to `test/rank_eval/YYYY-MM-DD_HH-MM-SS/results.json`.

**Validation**

Validated with focused output path tests in `tests/test_rank_eval_interface.py`, `python3 -m unittest discover tests`, `python3 -m py_compile rank_eval.py tests/test_rank_eval_interface.py`, and `python3 rank_eval.py --help`. Full rank evaluation was not run.

**Related files**

- `rank_eval.py`
- `tests/test_rank_eval_interface.py`

## `rank_eval.py` could not import SuperVLAD parser without `PYTHONPATH`

**Symptom**

`python3 rank_eval.py ...` failed with `ModuleNotFoundError: No module named 'parser'`.

**Context**

The failure happened when launching `rank_eval.py` from the repository root without first exporting `PYTHONPATH="$PWD:$PWD/third_party/SuperVLAD:${PYTHONPATH:-}"`.

**Likely cause**

`rank_eval.py` imported the vendored SuperVLAD `parser.py` module before adding `third_party/SuperVLAD` to `sys.path`.

**Fix or workaround**

`rank_eval.py` now resolves `third_party/SuperVLAD` relative to the script file and prepends it to `sys.path` before importing SuperVLAD modules.

**Validation**

Validated with `python3 rank_eval.py --help`, `python3 -m unittest discover tests`, `python3 tests/test_rank_eval_interface.py`, and `python3 -m py_compile rank_eval.py tests/test_rank_eval_interface.py`.

**Related files**

- `rank_eval.py`
- `tests/test_rank_eval_interface.py`

## NumPy removed `np.float` alias in SuperVLAD dataset loader

**Symptom**

`perceptual_eval.py` failed with `AttributeError: module 'numpy' has no attribute 'float'`.

**Context**

The failure occurred while `third_party/SuperVLAD/datasets_ws.py` loaded evaluation dataset UTM coordinates from database and query image filenames.

**Likely cause**

The repository pins `numpy==2.2.6`, and NumPy removed aliases such as `np.float` after deprecating them in NumPy 1.20.

**Fix or workaround**

Use the builtin `float` type for `.astype(...)` when converting parsed UTM coordinate strings.

**Validation**

Validated with `python3 -m py_compile third_party/SuperVLAD/datasets_ws.py` and a source search showing no remaining live `np.float` references in the relevant project Python files. Full dataset evaluation was not run.

**Related files**

- `third_party/SuperVLAD/datasets_ws.py`
- `perceptual_eval.py`
- `requirements.txt`

## Root README is empty

**Symptom**

The root `README.md` contains no setup, run, or project purpose content.

**Context**

Initial project investigation needed to rely on scripts, requirements, and third-party documentation.

**Likely cause**

Unknown.

**Fix or workaround**

Use `docs/` as the project documentation source for current engineering notes.

**Validation**

Observed during repository inspection.

**Related files**

- `README.md`
- `docs/`
