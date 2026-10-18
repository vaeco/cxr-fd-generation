#!/usr/bin/env python3
from __future__ import annotations

# Allow both python -m scripts.NAME and python scripts/NAME.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import CheXpertManifestDataset, make_balanced_loader
from src.fd_loss import (
    FeatureQueue,
    frechet_distance_loss,
    load_reference_stats,
    normalized_fd,
)
from src.models import (
    ConditionalGenerator,
    ProjectionDiscriminator,
    freeze_module,
    load_feature_classifier,
)
from src.utils import (
    AverageMeter,
    atomic_torch_save,
    count_parameters,
    ensure_dir,
    marker,
    optimizer_to,
    save_tensor_grid,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train common, Base, or Base+C-FD GAN")
    parser.add_argument("--mode", choices=["pretrain", "base", "fd"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--feature-ckpt", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, required=True, help="Epochs to add in this run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--lambda-fd", type=float, default=0.10)
    parser.add_argument("--fd-offset", type=float, default=0.01)
    parser.add_argument("--fd-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "z_dim": args.z_dim,
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "num_classes": 2,
    }


def validate_parent(parent: dict[str, Any], args: argparse.Namespace) -> None:
    expected = checkpoint_config(args)
    found = parent.get("config", {})
    mismatches = {
        key: (found.get(key), value)
        for key, value in expected.items()
        if found.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint architecture mismatch: {mismatches}")


@torch.no_grad()
def initialize_queues(
    generator: ConditionalGenerator,
    feature_model,
    queues: dict[int, FeatureQueue],
    z_dim: int,
    device: torch.device,
    batch_size: int,
    logger,
) -> None:
    was_training = generator.training
    generator.eval()
    for class_id, queue in queues.items():
        collected = []
        remaining = queue.size
        while remaining > 0:
            current = min(batch_size, remaining)
            labels = torch.full((current,), class_id, device=device, dtype=torch.long)
            noise = torch.randn(current, z_dim, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                images = generator(noise, labels)
            features = feature_model.extract_features(images.float())
            collected.append(features.detach())
            remaining -= current
        queue.initialize(torch.cat(collected, dim=0))
        marker(
            logger,
            "QUEUE_INIT",
            class_id=class_id,
            count=queue.size,
            feature_dim=queue.feature_dim,
            mean_norm=f"{queue.features.mean(0).norm().item():.4f}",
        )
    generator.train(was_training)


def save_curves(history: list[dict[str, float]], path: Path, with_fd: bool) -> None:
    frame = pd.DataFrame(history)
    columns = 3 if with_fd else 2
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 4))
    axes = list(axes) if columns > 1 else [axes]
    axes[0].plot(frame["epoch"], frame["loss_d"])
    axes[0].set_title("Discriminator hinge loss")
    axes[0].set_xlabel("Global epoch")
    axes[1].plot(frame["epoch"], frame["loss_g_total"], label="total")
    axes[1].plot(frame["epoch"], frame["loss_g_adv"], label="adversarial")
    axes[1].set_title("Generator loss")
    axes[1].set_xlabel("Global epoch")
    axes[1].legend()
    if with_fd:
        axes[2].plot(frame["epoch"], frame["loss_fd"], label="normalized C-FD")
        axes[2].plot(frame["epoch"], frame["raw_fd_c0"], label="raw FD class 0")
        axes[2].plot(frame["epoch"], frame["raw_fd_c1"], label="raw FD class 1")
        axes[2].plot(frame["epoch"], frame["lambda_fd"], label="lambda")
        axes[2].set_title("C-FD branch")
        axes[2].set_xlabel("Global epoch")
        axes[2].set_yscale("symlog", linthresh=0.05)
        axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.mode in {"base", "fd"} and args.resume is None:
        raise SystemExit("Base and FD branches must use --resume COMMON_CHECKPOINT")
    if args.mode == "fd" and (args.feature_ckpt is None or args.stats_dir is None):
        raise SystemExit("FD mode requires --feature-ckpt and --stats-dir")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GAN training is intended for CUDA")
    output_dir = ensure_dir(args.output_dir)
    samples_dir = ensure_dir(output_dir / "samples")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logger = setup_logger(f"gan_{args.mode}", output_dir / f"train_{args.mode}.log")
    writer = SummaryWriter(output_dir / "tensorboard")

    dataset = CheXpertManifestDataset(
        args.manifest, split="train", image_size=args.image_size, train=True
    )
    loader, sampler = make_balanced_loader(
        dataset, args.batch_size, args.workers, args.seed
    )
    generator = ConditionalGenerator(
        z_dim=args.z_dim,
        base_channels=args.base_channels,
        image_size=args.image_size,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        base_channels=args.base_channels, image_size=args.image_size
    ).to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )
    scaler_g = torch.amp.GradScaler("cuda", enabled=True)
    scaler_d = torch.amp.GradScaler("cuda", enabled=True)
    start_epoch = 0
    global_step = 0
    branch_step = 0
    parent_path = None

    fixed_generator = torch.Generator().manual_seed(args.seed + 999)
    fixed_noise = torch.randn(32, args.z_dim, generator=fixed_generator)
    fixed_labels = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
    parent: dict[str, Any] | None = None
    if args.resume is not None:
        parent_path = str(args.resume.resolve())
        parent = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_parent(parent, args)
        generator.load_state_dict(parent["generator"])
        discriminator.load_state_dict(parent["discriminator"])
        optimizer_g.load_state_dict(parent["optimizer_g"])
        optimizer_d.load_state_dict(parent["optimizer_d"])
        optimizer_to(optimizer_g, device)
        optimizer_to(optimizer_d, device)
        start_epoch = int(parent.get("epoch", 0))
        global_step = int(parent.get("global_step", 0))
        if "fixed_noise" in parent:
            fixed_noise = parent["fixed_noise"].cpu()
            fixed_labels = parent["fixed_labels"].cpu()
        if parent.get("mode") == args.mode:
            branch_step = int(parent.get("branch_step", 0))

    feature_model = None
    queues: dict[int, FeatureQueue] = {}
    references: dict[int, dict[str, Any]] = {}
    if args.mode == "fd":
        feature_model, _ = load_feature_classifier(args.feature_ckpt, device)
        freeze_module(feature_model)
        for class_id in (0, 1):
            references[class_id] = load_reference_stats(
                args.stats_dir / f"class_{class_id}.npz", device
            )
            queues[class_id] = FeatureQueue(
                size=args.queue_size, feature_dim=feature_model.feature_dim
            ).to(device)
        resumed_queues = parent is not None and "queues" in parent and parent.get("mode") == "fd"
        if resumed_queues:
            for class_id in (0, 1):
                queues[class_id].load_state_dict(parent["queues"][str(class_id)])
                marker(logger, "QUEUE_RESUME", class_id=class_id, count=args.queue_size)
        else:
            initialize_queues(
                generator,
                feature_model,
                queues,
                args.z_dim,
                device,
                batch_size=args.batch_size,
                logger=logger,
            )

    # Queue warm-start and feature-model construction must not shift a new branch's
    # adversarial noise stream. A same-mode interrupted run restores its saved RNG.
    same_mode_resume = parent is not None and parent.get("mode") == args.mode
    if same_mode_resume and "torch_rng_state" in parent:
        torch.set_rng_state(parent["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_states" in parent:
            torch.cuda.set_rng_state_all(parent["cuda_rng_states"])
        marker(logger, "RNG_RESTORE", reason="same_mode_resume")
    else:
        seed_everything(args.seed)
        marker(logger, "RNG_RESET", seed=args.seed, reason="fair_branch_noise_stream")

    end_epoch = start_epoch + args.epochs
    marker(
        logger,
        "GAN_START",
        mode=args.mode,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        steps_per_epoch=len(loader),
        batch_size=args.batch_size,
        generator_params=count_parameters(generator),
        discriminator_params=count_parameters(discriminator),
        parent=parent_path or "none",
        seed=args.seed,
    )
    if args.mode == "fd":
        marker(
            logger,
            "FD_CONFIG",
            lambda_fd=args.lambda_fd,
            queue_size=args.queue_size,
            feature_dim=feature_model.feature_dim,
            warmup_steps=args.fd_warmup_steps,
            class_conditional=True,
        )

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch + 1, end_epoch + 1):
        sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()
        meters = {
            "loss_d": AverageMeter(),
            "loss_g_total": AverageMeter(),
            "loss_g_adv": AverageMeter(),
            "loss_fd": AverageMeter(),
            "grad_g": AverageMeter(),
            "lambda_fd": AverageMeter(),
            "raw_fd_c0": AverageMeter(),
            "raw_fd_c1": AverageMeter(),
        }
        progress = tqdm(loader, desc=f"{args.mode} epoch {epoch}/{end_epoch}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            current_batch = images.shape[0]

            # Discriminator: hinge objective.
            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(current_batch, args.z_dim, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                fake_images = generator(noise, labels)
                real_scores = discriminator(images, labels)
                fake_scores = discriminator(fake_images.detach(), labels)
                loss_d = F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # Generator: identical adversarial path in both final branches.
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(current_batch, args.z_dim, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                generated = generator(noise, labels)
                loss_g_adv = -discriminator(generated, labels).mean()
            loss_fd = torch.zeros((), device=device)
            lambda_current = 0.0
            current_features = None
            raw_fd_values = {0: 0.0, 1: 0.0}
            if args.mode == "fd":
                # Do not use no_grad/detach here: FD must reach the generator.
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    current_features = feature_model.extract_features(generated.float())
                class_losses = []
                for class_id in (0, 1):
                    class_features = current_features[labels == class_id]
                    queue_mean, queue_covariance = queues[class_id].differentiable_stats(
                        class_features
                    )
                    reference = references[class_id]
                    raw_fd = frechet_distance_loss(
                        queue_mean,
                        queue_covariance,
                        reference["mean"],
                        reference["covariance"],
                        reference["sqrt"],
                    )
                    raw_fd_values[class_id] = float(raw_fd.detach().item())
                    class_losses.append(normalized_fd(raw_fd, args.fd_offset))
                loss_fd = torch.stack(class_losses).mean().float()
                if not torch.isfinite(loss_fd):
                    raise FloatingPointError(f"Non-finite FD loss at global_step={global_step}")
                ramp = min(1.0, (branch_step + 1) / max(args.fd_warmup_steps, 1))
                lambda_current = args.lambda_fd * ramp
            loss_g_total = loss_g_adv.float() + lambda_current * loss_fd
            scaler_g.scale(loss_g_total).backward()
            scaler_g.unscale_(optimizer_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            if args.mode == "fd":
                # Historical samples are updated only after backward/step.
                for class_id in (0, 1):
                    queues[class_id].enqueue(current_features[labels == class_id])

            global_step += 1
            branch_step += 1
            meters["loss_d"].update(float(loss_d.item()))
            meters["loss_g_total"].update(float(loss_g_total.item()))
            meters["loss_g_adv"].update(float(loss_g_adv.item()))
            meters["loss_fd"].update(float(loss_fd.item()))
            meters["grad_g"].update(float(grad_norm))
            meters["lambda_fd"].update(lambda_current)
            meters["raw_fd_c0"].update(raw_fd_values[0])
            meters["raw_fd_c1"].update(raw_fd_values[1])
            writer.add_scalar("step/loss_d", float(loss_d.item()), global_step)
            writer.add_scalar("step/loss_g_total", float(loss_g_total.item()), global_step)
            if args.mode == "fd":
                writer.add_scalar("step/loss_fd", float(loss_fd.item()), global_step)
                writer.add_scalar("step/lambda_fd", lambda_current, global_step)
            if global_step % args.log_every == 0:
                marker(
                    logger,
                    "TRAIN_STEP",
                    mode=args.mode,
                    epoch=epoch,
                    step=global_step,
                    loss_d=f"{meters['loss_d'].avg:.4f}",
                    loss_g=f"{meters['loss_g_total'].avg:.4f}",
                    loss_fd=f"{meters['loss_fd'].avg:.4f}",
                    raw_fd_c0=f"{meters['raw_fd_c0'].avg:.3f}",
                    raw_fd_c1=f"{meters['raw_fd_c1'].avg:.3f}",
                    grad_g=f"{meters['grad_g'].avg:.3f}",
                )
            progress.set_postfix(
                d=f"{meters['loss_d'].avg:.3f}",
                g=f"{meters['loss_g_total'].avg:.3f}",
                fd=f"{meters['loss_fd'].avg:.3f}",
            )

        row = {"epoch": epoch, **{key: meter.avg for key, meter in meters.items()}}
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(f"epoch/{key}", value, epoch)
        marker(
            logger,
            "EPOCH_SUMMARY",
            mode=args.mode,
            epoch=epoch,
            loss_d=f"{meters['loss_d'].avg:.5f}",
            loss_g=f"{meters['loss_g_total'].avg:.5f}",
            loss_adv=f"{meters['loss_g_adv'].avg:.5f}",
            loss_fd=f"{meters['loss_fd'].avg:.5f}",
            raw_fd_c0=f"{meters['raw_fd_c0'].avg:.4f}",
            raw_fd_c1=f"{meters['raw_fd_c1'].avg:.4f}",
            lambda_fd=f"{meters['lambda_fd'].avg:.5f}",
            grad_g=f"{meters['grad_g'].avg:.4f}",
        )

        if epoch % args.sample_every == 0 or epoch == end_epoch:
            generator.eval()
            with torch.inference_mode(), torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                sample = generator(fixed_noise.to(device), fixed_labels.to(device))
            sample_path = samples_dir / f"epoch_{epoch:04d}.png"
            save_tensor_grid(sample, sample_path, nrow=8)
            marker(logger, "SAMPLE_SAVED", epoch=epoch, file=sample_path)

        payload = {
            "mode": args.mode,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "branch_step": branch_step,
            "config": checkpoint_config(args),
            "fixed_noise": fixed_noise,
            "fixed_labels": fixed_labels,
            "manifest": str(args.manifest.resolve()),
            "parent_checkpoint": parent_path,
            "args": vars(args),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        if args.mode == "fd":
            payload["queues"] = {
                str(class_id): queue.state_dict() for class_id, queue in queues.items()
            }
        atomic_torch_save(payload, checkpoints_dir / "latest.pt")
        if epoch % args.save_every == 0 or epoch == end_epoch:
            epoch_path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
            atomic_torch_save(payload, epoch_path)
            marker(logger, "CHECKPOINT", epoch=epoch, file=epoch_path)

    final_names = {
        "pretrain": "common_pretrain_final.pt",
        "base": "base_final.pt",
        "fd": "cfd_final.pt",
    }
    final_path = output_dir / final_names[args.mode]
    atomic_torch_save(payload, final_path)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_curves(history, output_dir / "training_curves.png", with_fd=args.mode == "fd")
    writer.close()
    marker(
        logger,
        "TRAINING_DONE",
        mode=args.mode,
        final_epoch=end_epoch,
        global_step=global_step,
        checkpoint=final_path,
    )


if __name__ == "__main__":
    main()
