"""Frozen descriptor bank that supplies global hard negatives during training.

The bank holds one fp16 descriptor per training place, encoded once with the pretrained model.
Each query fetches its nearest entries from other places, which are appended to its in-batch negatives.
"""

import argparse
import dataclasses
import logging
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from src.config import amp_autocast
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


def encode_bank(model, loader, device: str, mixed_precision: bool) -> Tuple[Tensor, Tensor]:
    """Encode ``(images (B, 1, C, H, W), place_ids (B, 1))`` batches into fp16 ``(N, D)`` descriptors and ``(N,)`` ids."""
    model.eval()
    descriptors, place_ids = [], []
    with torch.inference_mode(), amp_autocast(mixed_precision, device):
        for images, ids in loader:
            descriptors.append(model(images[:, 0].to(device), queryflag=1).half().cpu())
            place_ids.append(ids[:, 0].long())
    return torch.cat(descriptors), torch.cat(place_ids)


def build_bank_dataset(args):
    """One un-augmented image per GSV-Cities training place, at the training resolution."""
    from torchvision import transforms as T

    from src import supervlad_compat
    from src.config import IMAGENET_MEAN_STD, TRAIN_CITIES
    from src.data import resolve_gsv_cities_base_path

    transform = T.Compose(
        [
            T.Resize(tuple(args.train_resize), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]
    )
    GSVCitiesDataset = supervlad_compat.import_gsv_cities_dataset()
    return GSVCitiesDataset(
        cities=TRAIN_CITIES,
        img_per_place=1,
        min_img_per_place=4,
        random_sample_from_each_place=False,
        transform=transform,
        base_path=f"{resolve_gsv_cities_base_path(args)}/",
    )


def main(argv=None) -> None:
    from torch.utils.data import DataLoader

    from src.cli import parse_arguments
    from src.models import get_model_adapter

    parser = argparse.ArgumentParser(description="Encode one image per training place into a descriptor bank file.")
    parser.add_argument("--out", required=True, help="bank file to write")
    parser.add_argument("--bank_batch_size", type=int, default=32, help="images per encoding batch")
    bank_args, remaining = parser.parse_known_args(argv)
    args = parse_arguments(remaining)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    adapter = get_model_adapter(args.model)
    model = adapter.build(args).model.to(args.device)
    adapter.load_weights(model, args.resume, args)
    loader = DataLoader(build_bank_dataset(args), batch_size=bank_args.bank_batch_size,
                        num_workers=min(4, args.num_workers), pin_memory=True)
    descriptors, place_ids = encode_bank(model, loader, args.device, args.mixed_precision)
    save_bank(bank_args.out, descriptors, place_ids, model=args.model, checkpoint=args.resume,
              image_size=list(args.train_resize))
    logging.info("Wrote %d descriptors of width %d to %s", descriptors.shape[0], descriptors.shape[1], bank_args.out)


if __name__ == "__main__":
    main()
