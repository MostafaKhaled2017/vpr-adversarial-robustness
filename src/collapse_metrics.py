"""Collapse and neighborhood-distortion monitoring (Task 2.4).

Listwise objectives can be satisfied degenerately: an encoder that maps everything to the
same point trivially puts every positive at rank 1 relative to a mined list it has also
destroyed. Recall on the validation set eventually catches this, but only after the run has
wasted its compute. These metrics catch it per validation pass.

Three signals, answering feedback line 53:

* **Variance collapse** — per-dimension standard deviation of the descriptor matrix. A
  dimension with zero spread carries no information.
* **Uniformity** — mean pairwise cosine similarity. Approaching 1 means every descriptor
  points the same way.
* **Neighborhood distortion** — ``knn_overlap``, the mean Jaccard overlap between each
  sample's k-nearest-neighbour set under the current embedding and under a fixed reference
  (the pretrained encoder). This is the one that speaks to retrieval directly: it asks how
  much the *ordering* the model induces has moved, independently of any recall threshold.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .config import amp_autocast


def _nearest_neighbour_indices(descriptors: Tensor, k: int) -> Tensor:
    """Indices of each sample's ``k`` nearest neighbours, excluding the sample itself."""
    distances = torch.cdist(descriptors, descriptors)
    sample_count = descriptors.shape[0]
    # Exclude self by pushing the diagonal to infinity rather than by slicing rank 0:
    # exact duplicates make "the first column is self" false.
    diagonal = torch.arange(sample_count, device=descriptors.device)
    distances[diagonal, diagonal] = float("inf")
    return torch.topk(distances, k=k, dim=1, largest=False).indices


def knn_overlap(current: Tensor, reference: Tensor, k: int = 10) -> float:
    """Mean Jaccard overlap of k-NN sets between two embeddings of the same samples.

    Returns ``nan`` when there are too few samples to have any neighbour at all.
    """
    if current.shape[0] != reference.shape[0]:
        raise ValueError("knn_overlap needs the same samples in both embeddings")

    sample_count = current.shape[0]
    effective_k = min(int(k), sample_count - 1)
    if effective_k < 1:
        return float("nan")

    current = current.detach().float()
    reference = reference.detach().float()
    current_neighbours = _nearest_neighbour_indices(current, effective_k)
    reference_neighbours = _nearest_neighbour_indices(reference, effective_k)

    membership = torch.zeros(sample_count, sample_count, dtype=torch.bool, device=current.device)
    rows = torch.arange(sample_count, device=current.device).unsqueeze(1)
    membership[rows, reference_neighbours] = True
    shared = membership[rows, current_neighbours].sum(dim=1).float()

    # Both sets have exactly effective_k members, so |A u B| = 2k - |A n B|.
    union = 2 * effective_k - shared
    return float((shared / union).mean().item())


def mean_pairwise_cosine(descriptors: Tensor) -> float:
    """Mean cosine similarity over distinct pairs; 1.0 means every direction coincides."""
    sample_count = descriptors.shape[0]
    if sample_count < 2:
        return float("nan")

    normalized = torch.nn.functional.normalize(descriptors.detach().float(), dim=1)
    similarity = normalized @ normalized.t()
    off_diagonal = ~torch.eye(sample_count, dtype=torch.bool, device=descriptors.device)
    return float(similarity[off_diagonal].mean().item())


def collapse_report(current: Tensor, reference: Tensor, k: int = 10) -> Dict[str, float]:
    """Collapse and neighborhood-distortion summary for one validation subset.

    ``current`` and ``reference`` must hold descriptors for the *same* samples in the same
    order — typically the current model and the pretrained model on a fixed subset.
    """
    if current.shape[0] != reference.shape[0]:
        raise ValueError("collapse_report needs the same samples in both embeddings")

    current = current.detach().float()
    dimension_std = current.std(dim=0, unbiased=False)

    return {
        "sample_count": int(current.shape[0]),
        "dimension_std_mean": float(dimension_std.mean().item()),
        "dimension_std_min": float(dimension_std.min().item()),
        "dimension_std_median": float(dimension_std.median().item()),
        "collapsed_dimension_fraction": float((dimension_std < 1e-6).float().mean().item()),
        "mean_pairwise_cosine": mean_pairwise_cosine(current),
        "knn_overlap": knn_overlap(current, reference, k),
    }


def reference_sample_indices(dataset, sample_count: int) -> Sequence[int]:
    """A fixed, deterministic subset of the validation set to monitor.

    Deterministic on purpose: the same images every epoch and every run, so k-NN overlap
    measures the embedding moving rather than the sample changing underneath it.
    """
    available = len(dataset)
    return list(range(min(int(sample_count), available)))


def collect_descriptors(
    model,
    dataset,
    indices: Sequence[int],
    batch_size: int,
    device,
    mixed_precision: bool = False,
) -> Tensor:
    """Descriptors for a fixed subset, on CPU so a reference can be held across epochs."""
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    collected = []
    try:
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with amp_autocast(mixed_precision, device):
                    descriptors = model(images, queryflag=0)
                collected.append(descriptors.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(collected, dim=0)
