from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .faiss_utils import create_flat_l2_index


@dataclass
class RetrievalAttackBatch:
    """Targets for one batch of retrieval attacks.

    ``positive_descriptors`` carries either the single hardest positive per query with
    shape ``(B, D)`` — the pre-Phase-2 representation, still produced by the evaluation
    paths — or the full padded positive set with shape ``(B, P, D)`` when multi-positive
    targeting is enabled. ``positive_mask`` marks the valid entries of the padded set and
    is ``None`` in the single-positive case. Consumers should read ``positive_bank`` and
    ``positive_bank_mask``, which normalise both representations to ``(B, P, D)`` and
    ``(B, P)``.
    """

    query_indices: Tensor
    clean_query_descriptors: Tensor
    positive_descriptors: Tensor
    negative_descriptors: Tensor
    positive_mask: Optional[Tensor] = None

    @property
    def positive_bank(self) -> Tensor:
        """Positive descriptors as ``(B, P, D)`` regardless of how they were stored."""
        if self.positive_descriptors.dim() == 2:
            return self.positive_descriptors.unsqueeze(1)
        return self.positive_descriptors

    @property
    def positive_bank_mask(self) -> Tensor:
        """Validity mask as ``(B, P)``; all-valid when no mask was supplied."""
        if self.positive_mask is not None:
            return self.positive_mask
        bank = self.positive_bank
        return torch.ones(bank.shape[:2], dtype=torch.bool, device=bank.device)

    @property
    def hardest_positive_descriptors(self) -> Tensor:
        """Legacy ``(B, D)`` view: the valid positive farthest from the clean query."""
        bank = self.positive_bank
        if bank.shape[1] == 1:
            return bank[:, 0, :]
        distances = torch.norm(bank - self.clean_query_descriptors.unsqueeze(1), p=2, dim=2)
        distances = distances.masked_fill(~self.positive_bank_mask, float("-inf"))
        hardest = distances.argmax(dim=1)
        return bank[torch.arange(bank.shape[0], device=bank.device), hardest, :]

    def subset(self, selector: Tensor) -> "RetrievalAttackBatch":
        return RetrievalAttackBatch(
            query_indices=self.query_indices[selector],
            clean_query_descriptors=self.clean_query_descriptors[selector],
            positive_descriptors=self.positive_descriptors[selector],
            negative_descriptors=self.negative_descriptors[selector],
            positive_mask=None if self.positive_mask is None else self.positive_mask[selector],
        )

    def __len__(self) -> int:
        return int(self.query_indices.shape[0])


def select_rank_targets(
    clean_descriptors: Tensor,
    place_ids: Tensor,
    adv_negatives: int,
    multi_positive: bool = False,
    image_mask: Optional[Tensor] = None,
) -> Optional[RetrievalAttackBatch]:
    """Mine attack targets from a batch of place-grouped descriptors.

    With ``multi_positive=False`` this keeps the single hardest positive per place, which
    is the behaviour every pre-Phase-2 run used. With ``multi_positive=True`` it keeps the
    full positive set per place, padded to the batch maximum and accompanied by a validity
    mask, so the attack objective can target the *closest* positive instead (Task 2.1).

    ``image_mask`` optionally marks which of the ``images_per_place`` slots hold real
    images, which is what makes places with differing positive counts representable.
    """
    batch_size, images_per_place, descriptor_dim = clean_descriptors.shape
    if images_per_place < 2:
        return None

    flat_descriptors = clean_descriptors.reshape(batch_size * images_per_place, descriptor_dim)
    flat_place_ids = place_ids.reshape(-1)
    query_descriptors = clean_descriptors[:, 0, :]

    if image_mask is None:
        flat_image_mask = torch.ones(batch_size * images_per_place, dtype=torch.bool, device=clean_descriptors.device)
    else:
        flat_image_mask = image_mask.reshape(-1).to(device=clean_descriptors.device, dtype=torch.bool)

    query_indices: List[int] = []
    clean_queries: List[Tensor] = []
    positive_sets: List[Tensor] = []
    negatives: List[Tensor] = []
    min_negatives = None

    for place_offset in range(batch_size):
        place_label = place_ids[place_offset, 0]
        candidate_mask = flat_image_mask.reshape(batch_size, images_per_place)[place_offset, 1:]
        positive_candidates = clean_descriptors[place_offset, 1:, :][candidate_mask]
        if positive_candidates.shape[0] == 0:
            continue

        # Negatives may never come from a padded slot of another place.
        negative_selector = (flat_place_ids != place_label) & flat_image_mask
        negative_candidates = flat_descriptors[negative_selector]
        if negative_candidates.shape[0] == 0:
            continue

        current_k = min(adv_negatives, negative_candidates.shape[0])
        negative_distances = torch.norm(
            negative_candidates - query_descriptors[place_offset].unsqueeze(0),
            p=2,
            dim=1,
        )
        hard_negative_indices = torch.topk(
            negative_distances,
            k=current_k,
            largest=False,
        ).indices

        if multi_positive:
            positive_sets.append(positive_candidates)
        else:
            positive_distances = torch.norm(
                positive_candidates - query_descriptors[place_offset].unsqueeze(0),
                p=2,
                dim=1,
            )
            hardest_positive_index = int(torch.argmax(positive_distances).item())
            positive_sets.append(positive_candidates[hardest_positive_index])

        query_indices.append(place_offset)
        clean_queries.append(query_descriptors[place_offset])
        negatives.append(negative_candidates[hard_negative_indices])
        min_negatives = current_k if min_negatives is None else min(min_negatives, current_k)

    if len(query_indices) == 0 or min_negatives is None or min_negatives == 0:
        return None

    trimmed_negatives = [negative[:min_negatives] for negative in negatives]
    device = clean_descriptors.device

    if multi_positive:
        positive_descriptors, positive_mask = pad_positive_sets(positive_sets, descriptor_dim, device)
    else:
        positive_descriptors = torch.stack(positive_sets, dim=0).detach()
        positive_mask = None

    return RetrievalAttackBatch(
        query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
        clean_query_descriptors=torch.stack(clean_queries, dim=0).detach(),
        positive_descriptors=positive_descriptors,
        negative_descriptors=torch.stack(trimmed_negatives, dim=0).detach(),
        positive_mask=positive_mask,
    )


def pad_positive_sets(
    positive_sets: List[Tensor],
    descriptor_dim: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Pad ragged per-query positive sets to ``(B, P_max, D)`` with a validity mask."""
    max_positives = max(int(positives.shape[0]) for positives in positive_sets)
    padded = torch.zeros(len(positive_sets), max_positives, descriptor_dim, device=device)
    mask = torch.zeros(len(positive_sets), max_positives, dtype=torch.bool, device=device)
    for row, positives in enumerate(positive_sets):
        count = int(positives.shape[0])
        padded[row, :count, :] = positives
        mask[row, :count] = True
    return padded.detach(), mask


def build_attack_targets(
    args,
    eval_ds,
    database_features: np.ndarray,
    clean_query_features: np.ndarray,
    limit_queries: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    positives_per_query = eval_ds.get_positives()
    valid_query_indices = np.flatnonzero(
        np.fromiter((len(positive_candidates) > 0 for positive_candidates in positives_per_query), dtype=bool)
    )
    if len(valid_query_indices) == 0:
        raise RuntimeError("No queries with positives were found, cannot run attack evaluation.")

    if limit_queries is not None:
        valid_query_indices = valid_query_indices[:limit_queries]

    database_features = np.ascontiguousarray(database_features.astype(np.float32, copy=False))
    query_features = np.ascontiguousarray(clean_query_features[valid_query_indices].astype(np.float32, copy=False))

    search_k = min(eval_ds.database_num, max(128, args.adv_negatives + 32))
    faiss_index = create_flat_l2_index(database_features.shape[1], args.device)
    faiss_index.add(database_features)
    _, ranked_neighbors = faiss_index.search(query_features, search_k)

    targets = []
    for target_offset, query_index in enumerate(valid_query_indices):
        positive_candidates = np.asarray(positives_per_query[query_index], dtype=np.int64)
        query_feature = query_features[target_offset]

        positive_distances = np.sum((database_features[positive_candidates] - query_feature[None, :]) ** 2, axis=1)
        positive_index = int(positive_candidates[np.argmin(positive_distances)])

        neighbor_candidates = ranked_neighbors[target_offset]
        negative_indexes = neighbor_candidates[~np.isin(neighbor_candidates, positive_candidates)]
        if len(negative_indexes) < args.adv_negatives:
            distances = np.sum((database_features - query_feature[None, :]) ** 2, axis=1)
            positive_mask = np.zeros(eval_ds.database_num, dtype=bool)
            positive_mask[positive_candidates] = True
            negative_candidates = np.flatnonzero(~positive_mask)
            negative_order = np.argsort(distances[negative_candidates])
            negative_indexes = negative_candidates[negative_order]

        negative_indexes = negative_indexes[: args.adv_negatives].astype(np.int64)
        targets.append(
            {
                "query_index": int(query_index),
                "positive_index": positive_index,
                "negative_indexes": negative_indexes,
            }
        )

    return targets, valid_query_indices
