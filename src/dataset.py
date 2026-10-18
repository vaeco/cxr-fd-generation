from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from .utils import worker_seed_fn


CLASS_NAMES = {0: "No Finding", 1: "Pleural Effusion"}


class PadResize:
    """Preserve anatomy aspect ratio, pad with black, then resize."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        side = max(image.size)
        image = ImageOps.pad(
            image,
            (side, side),
            method=Image.Resampling.BILINEAR,
            color=0,
            centering=(0.5, 0.5),
        )
        return image.resize((self.size, self.size), Image.Resampling.BILINEAR)


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    operations: list[object] = [PadResize(image_size)]
    if train:
        # No horizontal flip: laterality can be clinically meaningful.
        operations.append(
            transforms.RandomAffine(
                degrees=4,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=0,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(operations)


class CheXpertManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int = 128,
        train: bool = False,
        return_path: bool = False,
    ) -> None:
        frame = pd.read_csv(manifest)
        required = {"path", "class_id", "class_name", "patient_id", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No samples for split={split!r} in {manifest}")
        self.frame = frame
        self.transform = build_transform(image_size, train=train)
        self.return_path = return_path
        self.targets = frame["class_id"].astype(int).tolist()
        self.class_indices = {
            class_id: [index for index, target in enumerate(self.targets) if target == class_id]
            for class_id in sorted(set(self.targets))
        }
        if set(self.class_indices) != {0, 1}:
            raise ValueError(f"Expected classes 0 and 1, found {sorted(self.class_indices)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["path"])
        with Image.open(path) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        label = torch.tensor(int(row["class_id"]), dtype=torch.long)
        if self.return_path:
            return tensor, label, path
        return tensor, label


class BalancedBatchSampler(Sampler[list[int]]):
    """Return batches containing exactly half of each binary class."""

    def __init__(
        self,
        targets: Sequence[int],
        batch_size: int,
        seed: int = 2026,
        drop_last: bool = True,
    ) -> None:
        if batch_size % 2:
            raise ValueError("batch_size must be even for a balanced binary batch")
        self.class_indices = {
            0: torch.tensor([i for i, value in enumerate(targets) if int(value) == 0]),
            1: torch.tensor([i for i, value in enumerate(targets) if int(value) == 1]),
        }
        if min(len(indices) for indices in self.class_indices.values()) == 0:
            raise ValueError("Both classes must contain at least one sample")
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 2
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = bool(drop_last)
        longest = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = math.ceil(longest / self.per_class)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _cycled_indices(
        indices: torch.Tensor,
        needed: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        parts = []
        remaining = needed
        while remaining > 0:
            shuffled = indices[torch.randperm(len(indices), generator=generator)]
            take = min(remaining, len(shuffled))
            parts.append(shuffled[:take])
            remaining -= take
        return torch.cat(parts)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        needed = self.num_batches * self.per_class
        streams = {
            class_id: self._cycled_indices(indices, needed, generator)
            for class_id, indices in self.class_indices.items()
        }
        for batch_id in range(self.num_batches):
            start = batch_id * self.per_class
            stop = start + self.per_class
            batch = torch.cat([streams[0][start:stop], streams[1][start:stop]])
            order = torch.randperm(len(batch), generator=generator)
            yield batch[order].tolist()

    def __len__(self) -> int:
        return self.num_batches


def make_balanced_loader(
    dataset: CheXpertManifestDataset,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, BalancedBatchSampler]:
    sampler = BalancedBatchSampler(dataset.targets, batch_size=batch_size, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed_fn,
        generator=generator,
    )
    return loader, sampler


def make_sequential_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed_fn,
    )
