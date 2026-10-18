#!/usr/bin/env python3
from __future__ import annotations

# Allow both python -m scripts.NAME and python scripts/NAME.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import CheXpertManifestDataset
from src.models import freeze_module, load_feature_classifier
from src.utils import ensure_dir, marker, save_json, seed_everything, setup_logger, worker_seed_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute class-wise real CXR feature statistics")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("real_stats", output_dir / "compute_real_stats.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_feature_classifier(args.feature_ckpt, device)
    freeze_module(model)
    dataset = CheXpertManifestDataset(
        args.manifest, split="train", image_size=128, train=False
    )
    summary: dict[str, object] = {
        "feature_checkpoint": str(args.feature_ckpt.resolve()),
        "feature_dim": model.feature_dim,
        "classes": {},
    }
    for class_id in (0, 1):
        generator = torch.Generator().manual_seed(args.seed + class_id)
        all_indices = torch.tensor(dataset.class_indices[class_id])
        order = torch.randperm(len(all_indices), generator=generator)
        selected = all_indices[order[: min(args.max_per_class, len(all_indices))]].tolist()
        loader = DataLoader(
            Subset(dataset, selected),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=worker_seed_fn,
        )
        collected = []
        with torch.inference_mode():
            for images, _ in tqdm(loader, desc=f"real stats class {class_id}", leave=False):
                features = model.extract_features(images.to(device, non_blocking=True))
                collected.append(features.float().cpu().numpy())
        array = np.concatenate(collected, axis=0).astype(np.float64)
        mean = array.mean(axis=0)
        covariance = np.cov(array, rowvar=False)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = np.linalg.eigvalsh(covariance)
        path = output_dir / f"class_{class_id}.npz"
        np.savez_compressed(
            path,
            mean=mean,
            covariance=covariance,
            count=np.asarray(len(array), dtype=np.int64),
            class_id=np.asarray(class_id, dtype=np.int64),
        )
        class_summary = {
            "count": int(len(array)),
            "mean_norm": float(np.linalg.norm(mean)),
            "covariance_trace": float(np.trace(covariance)),
            "min_eigenvalue": float(eigenvalues.min()),
            "file": str(path.resolve()),
        }
        summary["classes"][str(class_id)] = class_summary
        marker(
            logger,
            "REF_STATS",
            class_id=class_id,
            count=len(array),
            mean_norm=f"{class_summary['mean_norm']:.4f}",
            cov_trace=f"{class_summary['covariance_trace']:.4f}",
            min_eig=f"{class_summary['min_eigenvalue']:.3e}",
            file=path,
        )
    save_json(summary, output_dir / "summary.json")
    marker(
        logger,
        "REF_STATS_READY",
        feature_epoch=checkpoint.get("epoch", "unknown"),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
