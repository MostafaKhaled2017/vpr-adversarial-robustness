# Known Issues

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
