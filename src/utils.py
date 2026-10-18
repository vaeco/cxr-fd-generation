from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def worker_seed_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def marker(logger: logging.Logger, name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    logger.info("[%s] %s", name, payload)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_tensor_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    padding: int = 2,
) -> None:
    """Save images whose values are in [-1, 1]."""
    path = Path(path)
    ensure_dir(path.parent)
    images = images.detach().float().cpu().clamp(-1, 1)
    images = (images + 1.0) / 2.0
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)
    save_image(grid, path)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def chunks(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    bucket: list[Any] = []
    for value in values:
        bucket.append(value)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
