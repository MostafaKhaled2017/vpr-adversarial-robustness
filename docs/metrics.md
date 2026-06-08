# Metrics

## Recall at K

The primary retrieval metric is recall at K. Parser defaults and local evaluators include `R@1`, `R@5`, `R@10`, and `R@100`.

## Rank displacement

`rank_eval.py` computes the nearest-positive rank before and after attack for valid attacked queries per dataset and model. Rank displacement is reported as attacked nearest-positive rank minus clean nearest-positive rank, with count, mean, median, max, and p95 summaries.

## Attack success

`rank_eval.py` reports clean-correct attack success for queries whose model-specific clean nearest-positive rank is 1, and all-valid attack success for all attacked queries with positives. A query is counted as attack-successful when the nearest positive moves below rank 1.

## Attack runtime and query counts

`perceptual_eval.py` records dataset results, runtimes, and query counts for attack evaluation. `rank_eval.py` also reports attack runtime per query and skipped queries without positives.

## Notes

Reusable metric helpers live in `perceptual_adv_training/retrieval_metrics.py`.
