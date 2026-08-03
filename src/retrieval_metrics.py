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


def prepare_distance_database(database_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    database = np.ascontiguousarray(database_features.astype(np.float32, copy=False))
    database_norms = np.sum(database * database, axis=1)
    return database, database_norms


def squared_l2_distance_chunk(
    database: np.ndarray,
    database_norms: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    queries = np.ascontiguousarray(query_features.astype(np.float32, copy=False))
    query_norms = np.sum(queries * queries, axis=1, keepdims=True)
    distances = query_norms + database_norms[None, :] - 2.0 * queries @ database.T
    return np.maximum(distances, 0.0)


def _distance_chunk(database_features: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    database, database_norms = prepare_distance_database(database_features)
    return squared_l2_distance_chunk(database, database_norms, query_features)


def nearest_positive_rank_from_distances(distances: np.ndarray, positive_indexes: Sequence[int]) -> int:
    positives = np.asarray(positive_indexes, dtype=np.int64)
    if positives.size == 0:
        return -1

    nearest_positive_distance = np.min(distances[positives])
    tied_indexes = np.flatnonzero(distances == nearest_positive_distance)
    if np.any(~np.isin(tied_indexes, positives)):
        order = np.argsort(distances)
        inverse_ranks = np.empty_like(order)
        inverse_ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
        return int(inverse_ranks[positives].min())
    return int(1 + np.count_nonzero(distances < nearest_positive_distance))


def nearest_positive_ranks(
    database_features: np.ndarray,
    query_features: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    chunk_size: int = 128,
) -> np.ndarray:
    ranks = np.full(len(positives_per_query), -1, dtype=np.int64)
    if len(positives_per_query) == 0:
        return ranks

    database, database_norms = prepare_distance_database(database_features)
    for start in range(0, len(positives_per_query), chunk_size):
        end = min(start + chunk_size, len(positives_per_query))
        distances = squared_l2_distance_chunk(database, database_norms, query_features[start:end])
        for local_index, positives in enumerate(positives_per_query[start:end]):
            ranks[start + local_index] = nearest_positive_rank_from_distances(
                distances[local_index],
                positives,
            )
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
    database, database_norms = prepare_distance_database(database_features)
    for start in range(0, len(positives_per_query), chunk_size):
        end = min(start + chunk_size, len(positives_per_query))
        distances = squared_l2_distance_chunk(database, database_norms, query_features[start:end])
        predictions = np.argsort(distances, axis=1)[:, :max_k]
        for local_index, pred in enumerate(predictions):
            positives = np.asarray(positives_per_query[start + local_index], dtype=np.int64)
            for recall_index, recall_value in enumerate(recall_values):
                if np.any(np.isin(pred[:recall_value], positives)):
                    recalls[recall_index:] += 1
                    break
    recalls = recalls / len(positives_per_query) * 100.0
    return format_recalls(recalls, recall_values)


DISPLACEMENT_CROSSING_BOUNDARIES = (1, 5, 10, 100)


def rank_displacement_summary(
    clean_ranks: np.ndarray,
    attacked_ranks: np.ndarray,
    database_size: int | None = None,
) -> Dict[str, object]:
    """Summarize how far attacks push the nearest positive down the ranking.

    Percentiles and the median are reported alongside the mean because a handful of
    queries pushed to the tail of the gallery dominate the mean. ``mean_normalized``
    divides the mean by the gallery size so displacements are comparable across
    datasets. ``p_cross_K`` is the fraction of valid queries whose nearest positive
    ranked at or above K before the attack and below K after it — the probability
    that the attack breaks R@K for a query it did not already break.
    """
    clean_ranks = np.asarray(clean_ranks)
    attacked_ranks = np.asarray(attacked_ranks)
    valid = (clean_ranks > 0) & (attacked_ranks > 0)
    valid_clean = clean_ranks[valid]
    valid_attacked = attacked_ranks[valid]
    displacement = valid_attacked - valid_clean
    resolved_database_size = int(database_size) if database_size else None

    if displacement.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "max": 0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "mean_normalized": None,
            "database_size": resolved_database_size,
            **{f"p_cross_{boundary}": 0.0 for boundary in DISPLACEMENT_CROSSING_BOUNDARIES},
        }

    mean_displacement = float(np.mean(displacement))
    crossings = {
        f"p_cross_{boundary}": float(
            np.count_nonzero((valid_clean <= boundary) & (valid_attacked > boundary)) / displacement.size
        )
        for boundary in DISPLACEMENT_CROSSING_BOUNDARIES
    }
    return {
        "count": int(displacement.size),
        "mean": mean_displacement,
        "median": float(np.median(displacement)),
        "max": int(np.max(displacement)),
        "p90": float(np.percentile(displacement, 90)),
        "p95": float(np.percentile(displacement, 95)),
        "p99": float(np.percentile(displacement, 99)),
        "mean_normalized": mean_displacement / resolved_database_size if resolved_database_size else None,
        "database_size": resolved_database_size,
        **crossings,
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


def per_query_rank_records(
    query_ids: Sequence[int],
    clean_ranks: np.ndarray,
    attacked_ranks: np.ndarray,
) -> list[Dict[str, object]]:
    """Expose the per-query ranks that the summary metrics are aggregated from.

    A rank of -1 means the query had no positive in the gallery and is excluded from
    every summary; such rows are still emitted so the release artifact stays complete.
    """
    clean = np.asarray(clean_ranks, dtype=np.int64)
    attacked = np.asarray(attacked_ranks, dtype=np.int64)
    if len(query_ids) != clean.size or clean.size != attacked.size:
        raise ValueError(
            "query_ids, clean_ranks, and attacked_ranks must have the same length, but received "
            f"{len(query_ids)}, {clean.size}, and {attacked.size}."
        )
    return [
        {
            "query_id": int(query_id),
            "clean_rank": int(clean_rank),
            "attacked_rank": int(attacked_rank),
            "clean_correct_at_1": bool(clean_rank == 1),
            "attacked_correct_at_1": bool(attacked_rank == 1),
        }
        for query_id, clean_rank, attacked_rank in zip(query_ids, clean, attacked)
    ]


def rank_metric_bundle(
    clean_ranks: np.ndarray,
    attacked_ranks: np.ndarray,
    database_size: int | None = None,
) -> Mapping[str, object]:
    return {
        "rank_displacement": rank_displacement_summary(clean_ranks, attacked_ranks, database_size),
        "attack_success": attack_success_metrics(clean_ranks, attacked_ranks),
    }
