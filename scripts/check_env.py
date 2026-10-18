#!/usr/bin/env python3
from __future__ import annotations

# Allow both python -m scripts.NAME and python scripts/NAME.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from src.utils import marker, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CXR-FD server environment")
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("environment")
    marker(
        logger,
        "ENV",
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
    )
    if not torch.cuda.is_available():
        raise SystemExit("[ENV_FAIL] CUDA is unavailable. Do not start training.")
    properties = torch.cuda.get_device_properties(0)
    marker(
        logger,
        "GPU",
        name=properties.name.replace(" ", "_"),
        memory_GB=f"{properties.total_memory / 1024**3:.1f}",
        capability=f"{properties.major}.{properties.minor}",
    )
    required = [
        "numpy",
        "pandas",
        "PIL",
        "scipy",
        "sklearn",
        "matplotlib",
        "tensorboard",
        "torchmetrics",
        "torchvision",
    ]
    versions = {}
    for name in required:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")
    marker(logger, "PACKAGES", **versions)
    if shutil.which("nvidia-smi"):
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        logger.info("[NVIDIA_SMI] %s", output)
    usage = shutil.disk_usage(Path.cwd())
    marker(logger, "DISK", free_GB=f"{usage.free / 1024**3:.1f}")
    if usage.free < 30 * 1024**3:
        logger.warning("[DISK_WARN] Less than 30 GB free; use the data disk.")
    if args.data_root is not None:
        train_csv = args.data_root / "train.csv"
        sample_dir = args.data_root / "train"
        marker(
            logger,
            "DATA_CHECK",
            root=args.data_root,
            train_csv=train_csv.exists(),
            train_dir=sample_dir.exists(),
        )
        if not train_csv.exists() or not sample_dir.exists():
            raise SystemExit("[ENV_FAIL] CheXpert data root is incomplete")
    marker(logger, "ENV_PASS", status="ready")


if __name__ == "__main__":
    main()

