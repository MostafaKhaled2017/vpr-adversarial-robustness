# Evaluation

## Clean evaluation

Clean retrieval evaluation uses SuperVLAD dataset loading and descriptor extraction, then reports recall metrics.

## FGSM evaluation

`fgsm_eval.py` evaluates clean results and configured FGSM epsilon values. It writes JSON reports and logs.

## Native rank attack evaluation

`rank_eval.py` evaluates query-only retrieval-native rank attacks against fixed clean database descriptor sets. It supports one or more datasets through `--datasets`, one or more checkpoints through `--models`, model labels through `--model_tags`, and attacks `rank_pgd_linf`, `rank_pgd_l2`, and `rank_apgd_linf`. Attacks are generated from the first listed model and evaluated on all listed models. The evaluator writes JSON and CSV reports under a timestamped run directory, such as `test/rank_eval/YYYY-MM-DD_HH-MM-SS/`, and records clean recall, attacked recall, nearest-positive rank displacement, attack success rates, query counts, and runtime.

The evaluator also supports opt-in implementation audit diagnostics through `--audit_attack_implementation` and optional `--audit_output_json`. Audit mode records normalized epsilon raw-pixel equivalents, denormalized adversarial min/max values, descriptor norm summaries, gradient norm summaries, loss-direction distance changes, and best-loss selection checks. The audit JSON is written inside the timestamped run directory and is also embedded in the main JSON report under `implementation_audit`.

For quick implementation checks, `--audit_sample_database_size` can be used with `--audit_attack_implementation` and `--max_queries` to extract only a small sampled gallery plus selected queries. This mode is audit-only and its recall/rank values are not benchmark-comparable.

Phase 2 instrumentation adds optional per-step trace output through `--trace_query_indices`, with traces written by default under `<run_dir>/traces/<dataset>/<attack>/<query_index>_eps_<epsilon>.csv`. Each trace row records loss, positive distance, hard-negative distance, nearest-positive rank, normalized/raw Linf perturbation norms, and whether the row became the best-so-far attack state.

The evaluator now writes per-query diagnostics for every attacked dataset/model/epsilon condition by default under `<run_dir>/diagnostics/`. Diagnostics include clean and attacked nearest-positive rank, attack success, clean and attacked positive/nearest-negative distances, margins, rank displacement, perturbation norm, descriptor-space `rho_q`, selected target indexes, and an estimated certified worst-case rank diagnostic. The CWR field is a descriptor-space sensitivity estimate, not a formal certificate. A Phase 2 report summarizing the five-query audit-sample trace/diagnostic run is available at `reports/sprint1/phase2_trace_diagnostics_results.md`.

Phase 3 reporting adds optional attack image artifact saving through `--save_attack_images`. When enabled, `rank_eval.py` writes selected clean query PNGs, attacked query PNGs, amplified signed perturbation PNGs, and absolute perturbation heatmap PNGs under `<run_dir>/attack_images/<dataset>/<attack>/eps_<epsilon>/`. Selection is limited by `--save_attack_image_count` and prioritizes `--trace_query_indices` when provided. A manifest is written to `<run_dir>/attack_images/attack_image_manifest.csv`.

`scripts/visualization.py` regenerates report figures from saved `rank_eval.py` CSV, diagnostics, trace, and optional attack-image outputs. It writes attack-strength, single-query trace, margin-vs-failure, and perturbation visibility figures under `reports/figures/supervisor_feedback/` by default. A Phase 3 report summarizing the image-saving smoke run and generated figures is available at `reports/sprint1/phase3_plot_image_grid_results.md`.

The final MSLS, SPED, and Nordland `rank_pgd_linf` evaluation is saved in `test/rank_eval/2026-06-08_18-46-59/`. A LaTeX/PDF summary report is available at `reports/native-rank-attacks-evaluation.tex` and `reports/native-rank-attacks-evaluation.pdf`. The report pins the main recall table at its source location in the compiled PDF.

## Perceptual attack evaluation

`perceptual_eval.py` compares a base checkpoint and a trained checkpoint on shared clean and perceptual attack data. Default perceptual attacks include `FastLagrangePerceptualAttack` and `PerceptualPGDAttack`.

## Validation status

Native rank attack unit tests, syntax checks, CLI help checks, and output path tests passed on 2026-06-08. The full `rank_pgd_linf` evaluation report was generated from the completed CSV on 2026-06-10 and compiled with `pdflatex`. The report PDF was regenerated with fixed table placement on 2026-06-11. Phase 1 audit unit tests, syntax checks, and CLI help checks passed on 2026-06-23. A Phase 1 audit sample run using 64 database images and 5 queries completed under `test/rank_eval/2026-06-24_10-44-33/`; the result summary is available at `reports/sprint1/phase1_native_rank_attack_audit_results.md`. Phase 2 trace/diagnostic unit tests, syntax checks, CLI help checks, and audit-sample smoke runs passed on 2026-06-24. The five-query Phase 2 output used for reporting is saved under `test/rank_eval/2026-06-24_17-06-38/`. Phase 3 figure-generation tests and a CSV-only figure smoke run passed on 2026-06-24. A Phase 3 image-saving audit-sample run completed under `test/rank_eval/2026-06-24_19-16-54/`, and the figure-generation script produced all expected PNGs under `reports/figures/supervisor_feedback_phase3_image_smoke/`, including the perturbation visibility grid.
