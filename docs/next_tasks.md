# Next Tasks

## High priority

- Run native rank attack evaluation on the intended MSLS setup.
  - Reason: Phase 1 implementation is available, but the full benchmark must be run to compare Rank-PGD-Linf against FGSM `training_style`.
  - Related files: `rank_eval.py`, `scripts/run_rank_eval.sh`, `fgsm_eval.py`

- Implement rank-margin diagnostics.
  - Reason: The next research direction needs positive margin and certified worst-case rank estimates to explain vulnerable queries.
  - Related files: `reports/vpr_student_research_decision_brief.md`, `reports/vpr_next_steps_implementation_plan.md`

- Rerun the original perceptual evaluation.
  - Reason: Confirm the NumPy compatibility fix against the real dataset and checkpoint setup.
  - Related files: `perceptual_eval.py`, `third_party/SuperVLAD/datasets_ws.py`

## Medium priority

- Implement retrieval-native perceptual rank attacks.
  - Reason: Current perceptual attacks are adapted through wrappers; the next stage needs FastLagrange-Rank and PerceptualPGD-Rank objectives.
  - Related files: `perceptual_adv_training/attacks.py`, `perceptual_eval.py`, `reports/vpr_next_steps_implementation_plan.md`

- Add a bounded geometric attack evaluator.
  - Reason: The supervisor brief asks whether geometry breaks VPR differently from pixel or perceptual attacks.
  - Related files: `reports/vpr_next_steps_implementation_plan.md`

- Add descriptor-level query-plus-database attack mode.
  - Reason: Query-only evaluation may overestimate robustness when database images or descriptors can also be perturbed.
  - Related files: `reports/vpr_next_steps_implementation_plan.md`

- Standardize interpreter usage in command examples.
  - Reason: `python` was not available in the active shell, while `python3` was available.
  - Related files: `commands.bash`, `installation.bash`

- Add lightweight smoke tests for dataset filename UTM parsing.
  - Reason: The dataset loader depends on filename format and currently has no root-level test command.
  - Related files: `third_party/SuperVLAD/datasets_ws.py`

## Low priority

- Inventory non-SuperVLAD VPR model options.
  - Reason: The benchmark should eventually show whether robustness behavior is SuperVLAD-specific or general across model families.
  - Related files: `reports/vpr_next_steps_implementation_plan.md`, `docs/models.md`

- Fill the root README or link it to `docs/`.
  - Reason: The root README is empty.
  - Related files: `README.md`, `docs/index.md`
