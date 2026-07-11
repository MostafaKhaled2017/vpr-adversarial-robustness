# Metrics

## Recall at K

The primary retrieval metric is recall at K. Parser defaults and local evaluators include `R@1`, `R@5`, `R@10`, and `R@100`.

## Rank displacement

`rank_eval.py` computes the nearest-positive rank before and after attack for valid attacked queries per dataset and model. Rank displacement is reported as attacked nearest-positive rank minus clean nearest-positive rank, with count, mean, median, max, and p95 summaries.

## Attack success

`rank_eval.py` reports clean-correct attack success for queries whose model-specific clean nearest-positive rank is 1, and all-valid attack success for all attacked queries with positives. A query is counted as attack-successful when the nearest positive moves below rank 1.

## Attack runtime and query counts

`perceptual_eval.py` records dataset results, runtimes, and query counts for attack evaluation. `rank_eval.py` also reports attack runtime per query and skipped queries without positives.

## Audit diagnostics

When `rank_eval.py` is run with `--audit_attack_implementation`, it records normalized epsilon raw-pixel equivalents, denormalized adversarial min/max values, clean and attacked descriptor L2 norm summaries, gradient norm summaries, positive and hard-negative distance changes, and best-loss selection checks. These diagnostics are intended to verify attack implementation behavior before running large experiment sweeps. When `--audit_sample_database_size` is used, recall and rank values are computed on a small sampled gallery and are not benchmark-comparable.

## Per-query diagnostics

`rank_eval.py` writes per-query diagnostics CSV files for attacked conditions. These files record clean and attacked nearest-positive rank, attack success, positive and nearest-negative descriptor distances, clean and attacked margins, rank displacement, perturbation norm, selected positive and hard-negative indexes, descriptor-space `rho_q`, and an estimated CWR value.

The estimated CWR uses measured descriptor movement from the clean and attacked query descriptors:

```text
rho_q = ||f(q_adv) - f(q_clean)||_2
CWR = 1 + count(n where max(d_clean(q,n) - rho_q, 0) <= d_clean(q,p) + rho_q)
```

This CWR value is a diagnostic sensitivity estimate and is not a formal robustness certificate.

## Trace metrics

When `--trace_query_indices` is supplied, per-step trace CSV rows include attack loss, positive distance, hard-negative distance, nearest-positive rank, normalized Linf perturbation norm, raw-pixel Linf perturbation norm, restart, step, epsilon, and a best-so-far flag.

## Figure outputs

`scripts/visualizations.py` uses rank summary CSVs, trace CSVs, diagnostics CSVs, and optional saved attack images to generate plots. The attack-strength plot uses attacked summary rows and a selected recall metric such as `R@1`. Trace plots use positive distance, hard-negative distance, and nearest-positive rank. The margin-vs-failure plot uses clean margin and attacked nearest-positive rank, with attack success as the outcome marker.

The Phase 3 image-saving smoke run under `test/rank_eval/2026-06-24_19-16-54/` generated all expected figure types, including the perturbation visibility grid. Its metrics are audit-sample-only and are not benchmark-comparable.

## Strength sweep summary

`scripts/run_rank_pgd_strength_sweep.py` collates Phase 5 per-condition JSON reports into `rank_pgd_strength_sweep_summary.csv`. The summary records clean and attacked R@1/5/10/100, clean-correct and all-valid attack success rates, mean and p95 rank displacement, runtime per query, mean perturbation norm, and the attack hyperparameters used for each dataset-condition result.

The runner selects the strongest practical setting using attacked `R@1` on the `base` model averaged across available datasets, then clean-correct attack success rate, runtime per query, and compute budget as tie-breakers.

When resuming a sweep, a job contributes to the summary only if its `rank_eval_results.json` passes validation for the expected dataset, model tags, attack condition, recalls, attack success, rank displacement, and perturbation metadata. Partial or interrupted outputs are rerun and are not included in the summary.

When `--max_dataset_samples` is omitted, sweep metrics use the full selected dataset. When `--max_dataset_samples` is set, metrics are computed on a deterministic sampled query/database gallery and are marked as not full-benchmark comparable; this mode is intended for comparing attack settings at lower cost.

The Sprint 1 attack-strength progress report uses the sweep summary CSVs plus per-query diagnostic CSVs to plot epsilon-baseline attacked R@1, clean versus attacked R@1 for the strongest tested setting, clean-correct attack success versus runtime, and clean/attacked margin shift.

## Notes

Reusable metric helpers live in `perceptual_adv_training/retrieval_metrics.py`.
