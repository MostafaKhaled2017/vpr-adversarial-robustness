"""Rank-PGD validation on a fixed random MSLS-val sample and clean-drop-constrained
checkpoint selection (spec D4)."""
import math
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import unwrap_model
from .rank_attacks import RankAttackConfig, RankPGDAttack
from .retrieval_metrics import compute_recalls_from_features
from .targets import build_attack_targets


def sample_validation_queries(positives_per_query: Sequence[Sequence[int]], count: int, seed: int) -> np.ndarray:
    """Sorted indices of ``count`` random queries that have positives (all of them if fewer)."""
    valid = np.flatnonzero([len(positives) > 0 for positives in positives_per_query])
    if len(valid) <= count:
        return valid
    return np.sort(np.random.default_rng(seed).choice(valid, size=count, replace=False))


def attack_condition_name(epsilon: float) -> str:
    return f"rank_pgd_linf_eps_{epsilon:g}"


def _extract(args, dataset, model: nn.Module, indices: Sequence[int], queryflag: int) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    features = []
    with torch.inference_mode():
        for inputs, _ in loader:
            features.append(model(inputs.to(args.device), queryflag=queryflag).float().cpu().numpy())
    return np.concatenate(features, axis=0)


def evaluate_rank_validation(args, model: nn.Module, val_ds, query_indices: np.ndarray) -> Dict[str, object]:
    """Clean and rank-PGD L-inf recalls on the sampled validation queries.

    Returns the same metric shape as ``evaluate_against_attacks_retrieval``: ``NoAttack``
    plus one ``rank_pgd_linf_eps_<eps>`` entry per epsilon (none for clean-only runs).
    """
    from .rank_eval import make_attack_batch

    query_indices = np.unique(np.asarray(query_indices, dtype=np.int64))
    descriptor_model = unwrap_model(model)
    was_training = descriptor_model.training
    descriptor_model.eval()
    try:
        val_ds.test_method = "hard_resize"
        database = _extract(args, val_ds, descriptor_model, range(val_ds.database_num), queryflag=0)
        val_ds.test_method = args.test_method
        queries = _extract(
            args, val_ds, descriptor_model, val_ds.database_num + query_indices, queryflag=1
        )
        positives_all = val_ds.get_positives()
        positives = [positives_all[index] for index in query_indices]
        metrics = {"NoAttack": compute_recalls_from_features(database, queries, positives, args.recall_values)}
        if args.is_clean_only:
            return metrics

        # build_attack_targets targets every query with positives, indexed by dataset query
        # id; hiding the positives of unsampled queries restricts it to the sample, in
        # sorted order matching ``query_indices``.
        sampled = set(query_indices.tolist())
        restricted_ds = SimpleNamespace(
            database_num=val_ds.database_num,
            get_positives=lambda: [
                positives_all[index] if index in sampled else [] for index in range(val_ds.queries_num)
            ],
        )
        full_queries = np.zeros((val_ds.queries_num, queries.shape[1]), dtype=np.float32)
        full_queries[query_indices] = queries
        targets, _ = build_attack_targets(args, restricted_ds, database, full_queries)
        database_tensor = torch.from_numpy(database).to(args.device)

        # One restart starting at the clean image: the attack is deterministic, so every
        # epoch is scored on the same attack without touching the training RNG stream.
        for epsilon in args.val_rank_epsilons:
            attack = RankPGDAttack(
                descriptor_model,
                RankAttackConfig(
                    epsilon=float(epsilon), steps=args.val_rank_steps, margin=args.adv_margin, device=args.device
                ),
            )
            attacked = []
            for offset in range(0, len(targets), args.infer_batch_size):
                query_inputs, attack_targets = make_attack_batch(
                    args, val_ds, targets[offset : offset + args.infer_batch_size], database_tensor, full_queries
                )
                result = attack(query_inputs, attack_targets)
                with torch.no_grad():
                    attacked.append(descriptor_model(result.adversarial, queryflag=1).float().cpu().numpy())
            metrics[attack_condition_name(epsilon)] = compute_recalls_from_features(
                database, np.concatenate(attacked, axis=0), positives, args.recall_values
            )
        return metrics
    finally:
        descriptor_model.train(was_training)


def select_checkpoint(
    metrics: Dict[str, object], initial_clean_r1: Optional[float], max_clean_drop: float, is_clean_only: bool
) -> Dict[str, object]:
    """Clean-drop-constrained selection: C = clean R@1, R = mean attacked R@1.

    Adversarial runs select on R among epochs whose C stays within ``max_clean_drop`` of the
    initial C (``initial_clean_r1=None`` marks the initial validation, eligible by
    definition); ineligible epochs score ``-inf`` so they never become best. Clean-only runs
    select on C.
    """
    clean = float(metrics["NoAttack"]["recalls"]["R@1"])
    if is_clean_only:
        return {"clean_score": clean, "robust_score": clean, "selection_score": clean, "eligible": True}
    attacked = [float(value["recalls"]["R@1"]) for name, value in metrics.items() if name != "NoAttack"]
    robust = float(np.mean(attacked))
    eligible = initial_clean_r1 is None or clean >= initial_clean_r1 - max_clean_drop
    return {
        "clean_score": clean,
        "robust_score": robust,
        "selection_score": robust if eligible else -math.inf,
        "eligible": bool(eligible),
    }
