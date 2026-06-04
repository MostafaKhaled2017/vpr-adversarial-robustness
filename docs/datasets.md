# Datasets

## Evaluation datasets

**Location**

Passed with `--eval_datasets_folder`; examples use `datasets`.

**Format**

Expected layout:

```text
<eval_datasets_folder>/<dataset_name>/images/<split>/database
<eval_datasets_folder>/<dataset_name>/images/<split>/queries
```

Image filenames must contain UTM coordinates in the form `@utm_easting@utm_northing@...@.jpg`.

**Purpose**

Validation, testing, and adversarial evaluation for visual place recognition.

**Notes**

Examples reference `msls`, `sped`, and `nordland`. Dataset contents are not included in the repository.

## GSV-Cities

**Location**

Passed with `--gsv_cities_base_path`, set with `GSV_CITIES_BASE_PATH`, or inferred as `<eval_datasets_folder>/gsv_cities`.

**Format**

Loaded by `third_party/SuperVLAD/dataloaders/train/GSVCitiesDataset.py`.

**Purpose**

Training data for adversarial fine-tuning.

**Notes**

Dataset files are not included in the repository.
