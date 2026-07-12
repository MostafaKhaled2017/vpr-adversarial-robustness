from typing import Dict, Mapping, Sequence

import numpy as np


def format_recalls(recalls: Sequence[float], recall_values: Sequence[int]) -> Dict[str, object]:
    recall_list = [float(recall) for recall in recalls]
    return {
        "recalls": {f"R@{value}": recall for value, recall in zip(recall_values, recall_list)},
        "recalls_list": recall_list,
        "recalls_str": ", ".join(
            [f"R@{value}: {recall:.1f}" for value, recall in zip(recall_values, recall_list)]
        ),
    }


def _distance_chunk(database_features: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    database = np.ascontiguousarray(database_features.astype(np.float32, copy=False))
    queries = np.ascontiguousarray(query_features.astype(np.float32, copy=False))
    query_norms = np.sum(queries * queries, axis=1, keepdims=True)
    database_norms = np.sum(database * database, axis=1)[None, :]
    distances = query_norms + database_norms - 2.0 * queries @ database.T
    return np.maximum(distances, 0.0)


def nearest_positive_ranks(
    database_features: np.ndarray,
    query_features: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    chunk_size: int = 128,
) -> np.ndarray:
    ranks = np.full(len(positives_per_query), -1, dtype=np.int64)
    if len(positives_per_query) == 0:
        return ranks

    for start in range(0, len(positives_per_query), chunk_size):
        end = min(start + chunk_size, len(positives_per_query))
        distances = _distance_chunk(database_features, query_features[start:end])
        order = np.argsort(distances, axis=1)
        inverse_ranks = np.empty_like(order)
        rank_values = np.arange(1, database_features.shape[0] + 1, dtype=np.int64)
        rows = np.arange(order.shape[0])[:, None]
        inverse_ranks[rows, order] = rank_values

        for local_index, positives in enumerate(positives_per_query[start:end]):
            positive_indexes = np.asarray(positives, dtype=np.int64)
            if positive_indexes.size == 0:
                continue
            ranks[start + local_index] = int(inverse_ranks[local_index, positive_indexes].min())
    return ranks


def compute_recalls_from_features(
    database_features: np.ndarray,
    query_features: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    recall_values: Sequence[int],
    chunk_size: int = 128,
) -> Dict[str, object]:
    if len(positives_per_query) == 0:
        return format_recalls([0.0 for _ in recall_values], recall_values)

    recalls = np.zeros(len(recall_values), dtype=np.float32)
    max_k = max(recall_values)
    for start in range(0, len(positives_per_query), chunk_size):
        end = min(start + chunk_size, len(positives_per_query))
        distances = _distance_chunk(database_features, query_features[start:end])
        predictions = np.argsort(distances, axis=1)[:, :max_k]
        for local_index, pred in enumerate(predictions):
            positives = np.asarray(positives_per_query[start + local_index], dtype=np.int64)
            for recall_index, recall_value in enumerate(recall_values):
                if np.any(np.isin(pred[:recall_value], positives)):
                    recalls[recall_index:] += 1
                    break
    recalls = recalls / len(positives_per_query) * 100.0
    return format_recalls(recalls, recall_values)


def rank_displacement_summary(clean_ranks: np.ndarray, attacked_ranks: np.ndarray) -> Dict[str, object]:
    valid = (clean_ranks > 0) & (attacked_ranks > 0)
    displacement = attacked_ranks[valid] - clean_ranks[valid]
    if displacement.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "max": 0,
            "p95": 0.0,
        }

    return {
        "count": int(displacement.size),
        "mean": float(np.mean(displacement)),
        "median": float(np.median(displacement)),
        "max": int(np.max(displacement)),
        "p95": float(np.percentile(displacement, 95)),
    }


def attack_success_metrics(clean_ranks: np.ndarray, attacked_ranks: np.ndarray) -> Dict[str, object]:
    valid = (clean_ranks > 0) & (attacked_ranks > 0)
    clean_correct = valid & (clean_ranks == 1)
    attacked_failure = valid & (attacked_ranks > 1)

    clean_correct_count = int(clean_correct.sum())
    valid_count = int(valid.sum())
    clean_correct_successes = int((clean_correct & attacked_failure).sum())
    all_valid_successes = int(attacked_failure.sum())

    return {
        "clean_correct": {
            "successes": clean_correct_successes,
            "total": clean_correct_count,
            "rate": float(clean_correct_successes / clean_correct_count * 100.0) if clean_correct_count else 0.0,
        },
        "all_valid": {
            "successes": all_valid_successes,
            "total": valid_count,
            "rate": float(all_valid_successes / valid_count * 100.0) if valid_count else 0.0,
        },
    }


def rank_metric_bundle(
    clean_ranks: np.ndarray,
    attacked_ranks: np.ndarray,
) -> Mapping[str, object]:
    return {
        "rank_displacement": rank_displacement_summary(clean_ranks, attacked_ranks),
        "attack_success": attack_success_metrics(clean_ranks, attacked_ranks),
    }
