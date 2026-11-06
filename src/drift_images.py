"""Unlabeled images of a folder, served as endless shuffled batches for the drift constraint."""
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import IMAGENET_MEAN_STD

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str | Path) -> List[Path]:
    """Every image file under ``folder``, recursively, in sorted order."""
    return sorted(path for path in Path(folder).rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


class FolderImages(Dataset):
    """The folder's images at ``image_size``, normalised as in training, without augmentation."""

    def __init__(self, folder: str | Path, image_size: Sequence[int]):
        from torchvision import transforms as T

        self.paths = list_images(folder)
        if not self.paths:
            raise ValueError(f"no image files under {folder}")
        self.transform = T.Compose(
            [
                T.Resize(tuple(image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _EndlessSampler(Sampler):
    """Image indices from ``position`` on, one seeded permutation per pass over the folder."""

    def __init__(self, size: int, seed: int, position: int):
        self.size, self.seed, self.position = size, seed, position

    def __iter__(self) -> Iterator[int]:
        position = self.position
        while True:
            permutation = np.random.default_rng((self.seed, position // self.size)).permutation(self.size)
            for offset in range(position % self.size, self.size):
                yield int(permutation[offset])
            position = (position // self.size + 1) * self.size


class DriftBatches:
    """Endless batches of a folder's images in a seeded order.

    The position (pass over the folder and offset in it) is saved with the training state, so
    a resume continues where the run stopped. Exact replay of an interrupted run is not a goal:
    batches the loader had prefetched but the run had not used are drawn again.
    """

    def __init__(
        self,
        folder: str | Path,
        image_size: Sequence[int],
        batch_size: int,
        seed: int,
        state: Optional[Dict[str, int]] = None,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        self.dataset = FolderImages(folder, image_size)
        self.batch_size = batch_size
        self.seed = seed
        size = len(self.dataset)
        self.position = 0 if not state else int(state["pass"]) * size + int(state["offset"])
        self._loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=_EndlessSampler(size, seed, self.position),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self._batches = None

    def __len__(self) -> int:
        return len(self.dataset)

    def next(self) -> Tensor:
        if self._batches is None:
            self._batches = iter(self._loader)
        batch = next(self._batches)
        self.position += len(batch)
        return batch

    def state_dict(self) -> Dict[str, int]:
        size = len(self.dataset)
        return {"pass": self.position // size, "offset": self.position % size}
