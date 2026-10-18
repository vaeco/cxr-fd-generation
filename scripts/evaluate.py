#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from src.dataset import CLASS_NAMES, CheXpertManifestDataset
from src.fd_loss import numpy_frechet_distance
from src.models import build_gan_from_checkpoint, freeze_module, load_feature_classifier
from src.utils import ensure_dir, marker, save_json, seed_everything, setup_logger, worker_seed_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair fixed-seed Base versus Base+C-FD evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-ckpt", type=Path, required=True)
    parser.add_argument("--cfd-ckpt", type=Path, required=True)
    parser.add_argument("--feature-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


@torch.no_grad()
def collect_real_features(
    feature_model,
    loader,
    device: torch.device,
) -> np.ndarray:
    collected = []
    for images, _ in loader:
        features = feature_model.extract_features(images.to(device, non_blocking=True))
        collected.append(features.float().cpu().numpy())
    return np.concatenate(collected, axis=0)


@torch.no_grad()
def evaluate_one(
    name: str,
    generator,
    z_dim: int,
    class_id: int,
    real_loader,
    real_features: np.ndarray,
    feature_model,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    kid_subsets: int,
    kid_subset_size: int,
    seed: int,
    logger,
) -> tuple[dict[str, float | int | str], torch.Tensor]:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(
        subsets=kid_subsets,
        subset_size=min(kid_subset_size, num_samples),
        normalize=True,
    ).to(device)
    for images, _ in real_loader:
        real = ((images.to(device, non_blocking=True) + 1.0) / 2.0).clamp(0, 1)
        real_rgb = real.repeat(1, 3, 1, 1)
        fid.update(real_rgb, real=True)
        kid.update(real_rgb, real=True)

    noise_generator = torch.Generator().manual_seed(seed + class_id * 1000)
    fake_features = []
    correct = 0
    generated_count = 0
    example_batch = None
    generator.eval()
    for start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"evaluate {name} class {class_id}",
        leave=False,
    ):
        current = min(batch_size, num_samples - start)
        noise = torch.randn(current, z_dim, generator=noise_generator).to(device)
        labels = torch.full((current,), class_id, device=device, dtype=torch.long)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            fake = generator(noise, labels)
        fake_float = fake.float()
        if example_batch is None:
            example_batch = fake_float[:8].cpu()
        fake_01 = ((fake_float + 1.0) / 2.0).clamp(0, 1)
        fake_rgb = fake_01.repeat(1, 3, 1, 1)
        fid.update(fake_rgb, real=False)
        kid.update(fake_rgb, real=False)
        medical_features = feature_model.extract_features(fake_float)
        fake_features.append(medical_features.float().cpu().numpy())
        predictions = feature_model.classifier(medical_features).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        generated_count += current
    fake_features_np = np.concatenate(fake_features, axis=0)
    fid_value = float(fid.compute().cpu())
    kid_mean, kid_std = kid.compute()
    row: dict[str, float | int | str] = {
        "model": name,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "n": generated_count,
        "fid": fid_value,
        "kid_mean": float(kid_mean.cpu()),
        "kid_std": float(kid_std.cpu()),
        "cxr_fd": numpy_frechet_distance(real_features, fake_features_np),
        "condition_accuracy": correct / generated_count,
    }
    marker(
        logger,
        "EVAL_CLASS",
        model=name,
        class_id=class_id,
        n=generated_count,
        fid=f"{row['fid']:.4f}",
        kid=f"{row['kid_mean']:.6f}",
        cxr_fd=f"{row['cxr_fd']:.4f}",
        cond_acc=f"{row['condition_accuracy']:.4f}",
    )
    return row, example_batch


def comparison_figure(examples: dict[tuple[str, int], torch.Tensor], path: Path) -> None:
    row_keys = [("Base", 0), ("Base+C-FD", 0), ("Base", 1), ("Base+C-FD", 1)]
    figure, axes = plt.subplots(4, 8, figsize=(14, 7.5))
    for row_index, key in enumerate(row_keys):
        model_name, class_id = key
        tensors = examples[key]
        for column in range(8):
            axes[row_index, column].imshow(
                ((tensors[column, 0].numpy() + 1) / 2).clip(0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{model_name}\n{CLASS_NAMES[class_id]}", rotation=90, fontsize=9
        )
    figure.suptitle("Fixed-noise qualitative comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def metric_figure(frame: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("fid", "FID ↓"),
        ("kid_mean", "KID ↓"),
        ("cxr_fd", "CXR-FD ↓"),
        ("condition_accuracy", "Condition accuracy ↑"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, title) in zip(axes.flat, metrics):
        pivot = frame.pivot(index="class_name", columns="model", values=column)
        pivot[["Base", "Base+C-FD"]].plot(kind="bar", ax=axis, rot=0)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.legend(title="")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=True)
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("evaluation", output_dir / "evaluate.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Evaluation is intended for CUDA")
    feature_model, _ = load_feature_classifier(args.feature_ckpt, device)
    freeze_module(feature_model)
    models = {
        "Base": build_gan_from_checkpoint(args.base_ckpt, device),
        "Base+C-FD": build_gan_from_checkpoint(args.cfd_ckpt, device),
    }
    base_config = models["Base"][1]["config"]
    cfd_config = models["Base+C-FD"][1]["config"]
    if base_config != cfd_config:
        raise ValueError("Base and C-FD checkpoints use different GAN architectures")
    if models["Base"][1].get("parent_checkpoint") != models["Base+C-FD"][1].get(
        "parent_checkpoint"
    ):
        raise ValueError("Final checkpoints do not share the same parent checkpoint")
    marker(
        logger,
        "FAIRNESS_CHECK",
        architecture_equal=True,
        parent_equal=True,
        base_epoch=models["Base"][1]["epoch"],
        cfd_epoch=models["Base+C-FD"][1]["epoch"],
        seed=args.seed,
    )
    if models["Base"][1]["epoch"] != models["Base+C-FD"][1]["epoch"]:
        raise ValueError("Base and C-FD final epochs differ")

    dataset = CheXpertManifestDataset(
        args.manifest, split="test", image_size=128, train=False
    )
    rows: list[dict[str, float | int | str]] = []
    examples: dict[tuple[str, int], torch.Tensor] = {}
    for class_id in (0, 1):
        indices = torch.tensor(dataset.class_indices[class_id])
        index_generator = torch.Generator().manual_seed(args.seed + class_id)
        selected = indices[
            torch.randperm(len(indices), generator=index_generator)[
                : min(args.num_per_class, len(indices))
            ]
        ].tolist()
        if len(selected) < 50:
            raise RuntimeError("At least 50 test images/class are required for FID/KID")
        subset = Subset(dataset, selected)
        real_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=worker_seed_fn,
        )
        real_features = collect_real_features(feature_model, real_loader, device)
        for model_name, (generator, checkpoint) in models.items():
            row, example = evaluate_one(
                name=model_name,
                generator=generator,
                z_dim=int(checkpoint["config"]["z_dim"]),
                class_id=class_id,
                real_loader=real_loader,
                real_features=real_features,
                feature_model=feature_model,
                device=device,
                num_samples=len(selected),
                batch_size=args.batch_size,
                kid_subsets=args.kid_subsets,
                kid_subset_size=args.kid_subset_size,
                seed=args.seed,
                logger=logger,
            )
            rows.append(row)
            examples[(model_name, class_id)] = example

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics_by_class.csv", index=False)
    macro = (
        frame.groupby("model")[["fid", "kid_mean", "cxr_fd", "condition_accuracy"]]
        .mean()
        .reset_index()
    )
    macro.to_csv(output_dir / "metrics_macro_average.csv", index=False)
    base = macro[macro["model"] == "Base"].iloc[0]
    cfd = macro[macro["model"] == "Base+C-FD"].iloc[0]
    delta = {
        "fid_reduction_percent": float((base["fid"] - cfd["fid"]) / base["fid"] * 100),
        "kid_reduction_percent": float(
            (base["kid_mean"] - cfd["kid_mean"]) / max(abs(base["kid_mean"]), 1e-12) * 100
        ),
        "cxr_fd_reduction_percent": float(
            (base["cxr_fd"] - cfd["cxr_fd"]) / base["cxr_fd"] * 100
        ),
        "condition_accuracy_change": float(
            cfd["condition_accuracy"] - base["condition_accuracy"]
        ),
    }
    save_json(delta, output_dir / "delta_summary.json")
    comparison_figure(examples, output_dir / "fixed_noise_comparison.png")
    metric_figure(frame, output_dir / "metric_comparison.png")
    logger.info("\nPer-class metrics:\n%s", frame.to_string(index=False))
    logger.info("\nMacro average:\n%s", macro.to_string(index=False))
    marker(
        logger,
        "EVAL_FINAL",
        fid_reduction_pct=f"{delta['fid_reduction_percent']:.2f}",
        kid_reduction_pct=f"{delta['kid_reduction_percent']:.2f}",
        cxr_fd_reduction_pct=f"{delta['cxr_fd_reduction_percent']:.2f}",
        condition_acc_change=f"{delta['condition_accuracy_change']:+.4f}",
    )
    marker(logger, "EVALUATION_DONE", output_dir=output_dir)


if __name__ == "__main__":
    main()
