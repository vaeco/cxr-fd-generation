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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import (
    CheXpertManifestDataset,
    make_balanced_loader,
    make_sequential_loader,
)
from src.models import CXRFeatureClassifier
from src.utils import (
    AverageMeter,
    atomic_torch_save,
    count_parameters,
    ensure_dir,
    marker,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen medical feature encoder")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: CXRFeatureClassifier,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        losses.update(float(loss.item()), images.shape[0])
        probabilities_all.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    predictions = (probabilities_np >= 0.5).astype(np.int64)
    return {
        "loss": losses.avg,
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "auc": float(roc_auc_score(labels_np, probabilities_np)),
    }


def save_history_plot(history: list[dict[str, float]], output_path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Classifier loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="accuracy")
    axes[1].plot(frame["epoch"], frame["val_auc"], label="AUC")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("classifier", output_dir / "train_classifier.log")
    writer = SummaryWriter(output_dir / "tensorboard")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Classifier training is intended for CUDA")

    train_dataset = CheXpertManifestDataset(
        args.manifest, split="train", image_size=224, train=True
    )
    val_dataset = CheXpertManifestDataset(
        args.manifest, split="val", image_size=224, train=False
    )
    test_dataset = CheXpertManifestDataset(
        args.manifest, split="test", image_size=224, train=False
    )
    train_loader, sampler = make_balanced_loader(
        train_dataset, args.batch_size, args.workers, args.seed
    )
    val_loader = make_sequential_loader(val_dataset, args.batch_size, args.workers)
    test_loader = make_sequential_loader(test_dataset, args.batch_size, args.workers)

    model = CXRFeatureClassifier(
        feature_dim=args.feature_dim, pretrained=args.pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    marker(
        logger,
        "CLASSIFIER_START",
        train=len(train_dataset),
        val=len(val_dataset),
        test=len(test_dataset),
        parameters=count_parameters(model),
        pretrained=args.pretrained,
    )

    history: list[dict[str, float]] = []
    best_auc = -1.0
    best_path = output_dir / "feature_encoder_best.pt"
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        train_loss = AverageMeter()
        progress = tqdm(train_loader, desc=f"classifier {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.update(float(loss.item()), images.shape[0])
            progress.set_postfix(loss=f"{train_loss.avg:.4f}")
        validation = evaluate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss.avg,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_auc": validation["auc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        marker(
            logger,
            "CLASSIFIER_EPOCH",
            epoch=epoch,
            train_loss=f"{train_loss.avg:.5f}",
            val_loss=f"{validation['loss']:.5f}",
            val_acc=f"{validation['accuracy']:.4f}",
            val_auc=f"{validation['auc']:.4f}",
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "feature_dim": args.feature_dim,
            "manifest": str(args.manifest),
            "val_metrics": validation,
            "args": vars(args),
        }
        atomic_torch_save(payload, output_dir / "feature_encoder_latest.pt")
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            atomic_torch_save(payload, best_path)
            marker(logger, "CLASSIFIER_BEST", epoch=epoch, val_auc=f"{best_auc:.4f}")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_history_plot(history, output_dir / "classifier_curves.png")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device)
    marker(
        logger,
        "CLASSIFIER_FINAL",
        best_epoch=best["epoch"],
        test_loss=f"{test_metrics['loss']:.5f}",
        test_acc=f"{test_metrics['accuracy']:.4f}",
        test_auc=f"{test_metrics['auc']:.4f}",
        checkpoint=best_path,
    )
    writer.close()


if __name__ == "__main__":
    main()
