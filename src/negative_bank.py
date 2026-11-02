"""Frozen descriptor bank that supplies global hard negatives during training.

The bank holds one fp16 descriptor per training place, encoded once with the pretrained model.
Each query fetches its nearest entries from other places, which are appended to its in-batch negatives.
"""

import dataclasses
import os
from typing import Optional

import torch
from torch import Tensor

from src.targets import RetrievalAttackBatch

BANK_FILE_KEYS = ("descriptors", "place_ids", "model", "checkpoint", "image_size")


def save_bank(path: str, descriptors: Tensor, place_ids: Tensor, **metadata) -> None:
    """Write the bank as fp16 descriptors and int64 place ids plus the given metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"descriptors": descriptors.half().cpu(), "place_ids": place_ids.long().cpu(), **metadata}, path)


# ponytail: the bank is frozen and never re-encoded; re-encode every few epochs if staleness shows.
class NegativeBank:
    """Nearest-neighbour lookup over a fixed ``(N, D)`` descriptor bank with ``(N,)`` place ids."""

    def __init__(self, descriptors: Tensor, place_ids: Tensor, device: str = "cuda", chunk_rows: int = 2048):
        self.descriptors = descriptors.to(device)
        self.place_ids = place_ids.long().to(device)
        self.chunk_rows = chunk_rows

    @classmethod
    def load(cls, path: str, device: str, chunk_rows: int = 2048, expected_dim: Optional[int] = None) -> "NegativeBank":
        state = torch.load(path, map_location="cpu", weights_only=True)
        width = state["descriptors"].shape[1]
        if expected_dim is not None and width != expected_dim:
            raise ValueError(f"bank descriptor width {width} does not match model descriptor width {expected_dim}")
        return cls(state["descriptors"], state["place_ids"], device=device, chunk_rows=chunk_rows)

    @property
    def descriptor_dim(self) -> int:
        return self.descriptors.shape[1]

    def __len__(self) -> int:
        return self.descriptors.shape[0]

    @torch.no_grad()
    def nearest(self, queries: Tensor, k: int, exclude_place_ids: Optional[Tensor] = None) -> Tensor:
        """Return the ``k`` bank descriptors closest to each query as ``(B, k, D)`` float32 on ``queries.device``.

        Rows whose place id equals the query's entry in ``exclude_place_ids`` are never returned
        while enough other rows remain. The bank is cast to fp32 one chunk at a time.
        """
        q = queries.detach().float()
        q_sq = q.pow(2).sum(1, keepdim=True)
        dist = torch.empty(q.shape[0], len(self), device=q.device)
        for start in range(0, len(self), self.chunk_rows):
            chunk = self.descriptors[start : start + self.chunk_rows].to(q.device).float()
            dist[:, start : start + chunk.shape[0]] = q_sq + chunk.pow(2).sum(1) - 2 * q @ chunk.T
        if exclude_place_ids is not None:
            excluded = self.place_ids.to(q.device)[None, :] == exclude_place_ids.to(q.device)[:, None]
            dist.masked_fill_(excluded, float("inf"))
        idx = dist.topk(k, dim=1, largest=False).indices
        return self.descriptors[idx.to(self.descriptors.device)].to(q.device).float()


def merge_bank_negatives(targets: RetrievalAttackBatch, extra: Tensor) -> RetrievalAttackBatch:
    """Append ``(B, K, D)`` bank negatives to the batch's ``(B, N, D)`` negatives."""
    negatives = torch.cat([targets.negative_descriptors, extra.to(targets.negative_descriptors)], dim=1)
    return dataclasses.replace(targets, negative_descriptors=negatives)
