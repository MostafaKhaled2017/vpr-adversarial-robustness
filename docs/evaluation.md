# Evaluation

## Clean evaluation

Clean retrieval evaluation uses SuperVLAD dataset loading and descriptor extraction, then reports recall metrics.

## FGSM evaluation

`fgsm_eval.py` evaluates clean results and configured FGSM epsilon values. It writes JSON reports and logs.

## Perceptual attack evaluation

`perceptual_eval.py` compares a base checkpoint and a trained checkpoint on shared clean and perceptual attack data. Default perceptual attacks include `FastLagrangePerceptualAttack` and `PerceptualPGDAttack`.

## Validation status

Only syntax validation for the touched dataset loader was run during the NumPy compatibility fix. Full evaluation was not run.
