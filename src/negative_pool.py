"""Cross-batch refreshed negative pool (Task 2.2).

A training batch of 16 places supplies at most a few dozen candidate negatives, so the
"hard" negatives an attack is asked to beat are only hard relative to a tiny sample. The
supervisor's feedback (line 51) asks for a larger or dynamically refreshed negative set.

This is a MoCo-style FIFO queue of detached descriptors tagged with their place IDs.
Descriptors go stale as the encoder moves, and the pool size bounds that staleness: an
entry survives at most ``capacity / batch_descriptors_per_step`` steps. Mining draws from
the pool *and* the current batch, so the freshest candidates are always in play.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class NegativePool:
    """FIFO queue of descriptors used to mine cross-place negatives.

    ``capacity=0`` disables the pool; ``push`` becomes a no-op and ``mine`` falls back to
    whatever the caller passes as the current batch.
    """

    def __init__(self, capacity: int, descriptor_dim: int, device: torch.device | str = "cpu"):
        if capacity < 0:
            raise ValueError("NegativePool capacity must be non-negative")
        self.capacity = int(capacity)
        self.descriptor_dim = int(descriptor_dim)
        self.device = torch.device(device)
        self._descriptors = torch.zeros(self.capacity, self.descriptor_dim, device=self.device)
        self._place_ids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self._write_cursor = 0
        self._size = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def __len__(self) -> int:
        return self._size

    def descriptors(self) -> Tensor:
        """The live entries, oldest-to-newest ordering not guaranteed."""
        return self._descriptors[: self._size]

    def place_ids(self) -> Tensor:
        return self._place_ids[: self._size]

    def push(self, descriptors: Tensor, place_ids: Tensor) -> None:
        """Append a step's descriptors, evicting the oldest entries when full."""
        if not self.enabled:
            return

        descriptors = descriptors.detach().to(device=self.device, dtype=self._descriptors.dtype)
        place_ids = place_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        if descriptors.shape[0] != place_ids.shape[0]:
            raise ValueError("push() needs one place id per descriptor")

        # A push larger than the queue would wrap onto itself; only the newest entries
        # would survive anyway, so slice to them up front.
        if descriptors.shape[0] > self.capacity:
            descriptors = descriptors[-self.capacity :]
            place_ids = place_ids[-self.capacity :]

        count = descriptors.shape[0]
        if count == 0:
            return

        end = self._write_cursor + count
        if end <= self.capacity:
            self._descriptors[self._write_cursor : end] = descriptors
            self._place_ids[self._write_cursor : end] = place_ids
        else:
            split = self.capacity - self._write_cursor
            self._descriptors[self._write_cursor :] = descriptors[:split]
            self._place_ids[self._write_cursor :] = place_ids[:split]
            self._descriptors[: count - split] = descriptors[split:]
            self._place_ids[: count - split] = place_ids[split:]

        self._write_cursor = end % self.capacity
        self._size = min(self.capacity, self._size + count)

    def mine(
        self,
        query_descriptors: Tensor,
        query_place_ids: Tensor,
        k: int,
        batch_descriptors: Optional[Tensor] = None,
        batch_place_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Return the ``k`` nearest cross-place negatives per query as ``(B, k', D)``.

        Candidates are the pool plus any current-batch descriptors supplied by the caller.
        ``k'`` is capped at the smallest number of candidates available to *any* query in
        the batch, because the result is a single stacked tensor and per-query counts
        cannot vary. Returns ``None`` when some query has no cross-place candidate at all.
        """
        candidate_descriptors, candidate_place_ids = self._candidates(batch_descriptors, batch_place_ids)
        if candidate_descriptors is None:
            return None

        query_descriptors = query_descriptors.detach().to(device=candidate_descriptors.device)
        query_place_ids = query_place_ids.detach().to(device=candidate_descriptors.device, dtype=torch.long).reshape(-1)

        cross_place = query_place_ids.unsqueeze(1) != candidate_place_ids.unsqueeze(0)  # (B, C)
        available = int(cross_place.sum(dim=1).min().item())
        if available == 0:
            return None

        effective_k = min(int(k), available)
        distances = torch.cdist(query_descriptors, candidate_descriptors)
        distances = distances.masked_fill(~cross_place, float("inf"))
        nearest = torch.topk(distances, k=effective_k, dim=1, largest=False).indices  # (B, k')
        return candidate_descriptors[nearest].detach()

    def _candidates(
        self,
        batch_descriptors: Optional[Tensor],
        batch_place_ids: Optional[Tensor],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        parts = []
        if self._size > 0:
            parts.append((self.descriptors(), self.place_ids()))
        if batch_descriptors is not None and batch_descriptors.shape[0] > 0:
            if batch_place_ids is None:
                raise ValueError("batch_place_ids is required when batch_descriptors is given")
            device = self.device if self._size > 0 else batch_descriptors.device
            parts.append(
                (
                    batch_descriptors.detach().to(device=device),
                    batch_place_ids.detach().to(device=device, dtype=torch.long).reshape(-1),
                )
            )
        if not parts:
            return None, None

        descriptors = torch.cat([part[0] for part in parts], dim=0)
        place_ids = torch.cat([part[1] for part in parts], dim=0)
        return descriptors, place_ids
