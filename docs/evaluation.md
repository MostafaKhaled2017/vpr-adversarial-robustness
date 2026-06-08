# Evaluation

## Clean evaluation

Clean retrieval evaluation uses SuperVLAD dataset loading and descriptor extraction, then reports recall metrics.

## FGSM evaluation

`fgsm_eval.py` evaluates clean results and configured FGSM epsilon values. It writes JSON reports and logs.

## Native rank attack evaluation

`rank_eval.py` evaluates query-only retrieval-native rank attacks against fixed clean database descriptor sets. It supports one or more datasets through `--datasets`, one or more checkpoints through `--models`, model labels through `--model_tags`, and attacks `rank_pgd_linf`, `rank_pgd_l2`, and `rank_apgd_linf`. Attacks are generated from the first listed model and evaluated on all listed models. The evaluator writes JSON and CSV reports under a timestamped run directory, such as `test/rank_eval/YYYY-MM-DD_HH-MM-SS/`, and records clean recall, attacked recall, nearest-positive rank displacement, attack success rates, query counts, and runtime.

## Perceptual attack evaluation

`perceptual_eval.py` compares a base checkpoint and a trained checkpoint on shared clean and perceptual attack data. Default perceptual attacks include `FastLagrangePerceptualAttack` and `PerceptualPGDAttack`.

## Validation status

Native rank attack unit tests, syntax checks, CLI help checks, and output path tests passed on 2026-06-08. Full rank evaluation was not run because it would require a full descriptor extraction/evaluation pass over the dataset.
