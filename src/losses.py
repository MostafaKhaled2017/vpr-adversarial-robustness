from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

LOSS_FN = None
MINER = None


def configure_metric_learning() -> None:
    global LOSS_FN, MINER

    from pytorch_metric_learning import losses, miners
    from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

    LOSS_FN = losses.MultiSimilarityLoss(
        alpha=1.0,
        beta=50,
        base=0.0,
        distance=DotProductSimilarity(),
    )
    MINER = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())


def loss_function(descriptors: Tensor, labels: Tensor) -> Tensor:
    if MINER is not None:
        miner_outputs = MINER(descriptors, labels)
        return LOSS_FN(descriptors, labels, miner_outputs)
    return LOSS_FN(descriptors, labels)


def as_positive_bank(positive_descriptor: Tensor) -> Tensor:
    """Normalise ``(B, D)`` or ``(B, P, D)`` positives to ``(B, P, D)``."""
    if positive_descriptor.dim() == 2:
        return positive_descriptor.unsqueeze(1)
    return positive_descriptor


def closest_positive_distance(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Distance to the nearest *valid* positive (Task 2.1).

    Retrieval succeeds when any positive outranks every negative, so the attack has to
    beat the closest positive rather than the single hardest one. With one positive per
    query this is exactly the pre-Phase-2 quantity.
    """
    bank = as_positive_bank(positive_descriptor)
    distances = torch.norm(query_descriptor.unsqueeze(1) - bank, p=2, dim=2)
    if positive_mask is not None:
        distances = distances.masked_fill(~positive_mask.to(torch.bool), float("inf"))
    return distances.min(dim=1).values


def compute_attack_score(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    margin: float,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    positive_distance = closest_positive_distance(query_descriptor, positive_descriptor, positive_mask)
    negative_distance = torch.norm(query_descriptor.unsqueeze(1) - negative_descriptors, p=2, dim=2)
    hardest_negative_distance = negative_distance.min(dim=1).values
    return margin + positive_distance - hardest_negative_distance


def compute_rank_loss(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    margin: float,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    return torch.relu(
        compute_attack_score(query_descriptor, positive_descriptor, negative_descriptors, margin, positive_mask)
    ).mean()


DEFAULT_LISTWISE_TAU = 0.05
DEFAULT_LISTWISE_K = 1
# Half-integer offset between "rank k is acceptable" and "rank k+1 is a violation". Without
# it the hinge sits exactly on the boundary and softplus(0) = ln 2 would charge a constant
# penalty to a query that already retrieves correctly at rank k.
LISTWISE_RANK_MARGIN = 0.5


def compute_soft_ranks(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    tau: float = DEFAULT_LISTWISE_TAU,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Smooth rank of each positive: 1 + a soft count of the negatives ranked above it.

    As ``tau -> 0`` the sigmoid becomes a step function and this converges to the true
    integer rank of the positive in the mined list. Masked positives get ``inf`` so they
    can never be the best.
    """
    bank = as_positive_bank(positive_descriptor)
    positive_distance = torch.norm(query_descriptor.unsqueeze(1) - bank, p=2, dim=2)  # (B, P)
    negative_distance = torch.norm(query_descriptor.unsqueeze(1) - negative_descriptors, p=2, dim=2)  # (B, N)
    difference = positive_distance.unsqueeze(2) - negative_distance.unsqueeze(1)  # (B, P, N)
    soft_rank = 1.0 + torch.sigmoid(difference / tau).sum(dim=2)
    if positive_mask is not None:
        soft_rank = soft_rank.masked_fill(~positive_mask.to(torch.bool), float("inf"))
    return soft_rank


def compute_listwise_scores(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    tau: float = DEFAULT_LISTWISE_TAU,
    k: int = DEFAULT_LISTWISE_K,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Per-query smooth surrogate of Recall@k under the mined list (Task 2.3).

    Retrieval is correct exactly when the best positive's rank is at most ``k``, so the
    objective penalises the best positive's *soft* rank exceeding ``k``. This is the
    Smooth-AP family (Brown et al., ECCV 2020); the difference is that it targets top-K
    recall over the mined list rather than average precision, and is evaluated on
    adversarial rather than clean queries.
    """
    soft_rank = compute_soft_ranks(query_descriptor, positive_descriptor, negative_descriptors, tau, positive_mask)
    best_soft_rank = soft_rank.min(dim=1).values
    return torch.nn.functional.softplus((best_soft_rank - k - LISTWISE_RANK_MARGIN) / tau)


def compute_listwise_loss(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    tau: float = DEFAULT_LISTWISE_TAU,
    k: int = DEFAULT_LISTWISE_K,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    return compute_listwise_scores(
        query_descriptor,
        positive_descriptor,
        negative_descriptors,
        tau,
        k,
        positive_mask,
    ).mean()


def compute_align_loss(clean_query_descriptor: Tensor, adv_query_descriptor: Tensor) -> Tensor:
    return (adv_query_descriptor - clean_query_descriptor).pow(2).sum(dim=1).mean()


def query_is_correct(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    positive_distance = closest_positive_distance(query_descriptor, positive_descriptor, positive_mask)
    negative_distance = torch.norm(query_descriptor.unsqueeze(1) - negative_descriptors, p=2, dim=2)
    return positive_distance < negative_distance.min(dim=1).values
