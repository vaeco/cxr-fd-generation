#!/usr/bin/env python3
from __future__ import annotations

# Allow both python -m scripts.NAME and python scripts/NAME.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataset import CheXpertManifestDataset
from src.utils import ensure_dir, marker, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the prepared CheXpert subset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("visualize_data", output_dir / "visualize_data.log")
    frame = pd.read_csv(args.manifest)
    train = CheXpertManifestDataset(args.manifest, split="train", image_size=128, train=False)
    rng = np.random.default_rng(args.seed)

    figure, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for row, class_id in enumerate((0, 1)):
        indices = np.asarray(train.class_indices[class_id])
        chosen = rng.choice(indices, size=min(8, len(indices)), replace=False)
        for column, index in enumerate(chosen):
            image, _ = train[int(index)]
            axes[row, column].imshow(((image[0].numpy() + 1) / 2), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_title("No Finding" if class_id == 0 else "Pleural Effusion")
    figure.suptitle("Patient-disjoint CheXpert training examples")
    figure.tight_layout()
    grid_path = output_dir / "data_examples.png"
    figure.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    count_frame = frame.groupby(["split", "class_name"]).size().reset_index(name="images")
    sns.barplot(data=count_frame, x="split", y="images", hue="class_name", ax=axes[0])
    axes[0].set_title("Balanced images per split")
    axes[0].set_xlabel("")
    brightness = {0: [], 1: []}
    for class_id in (0, 1):
        indices = train.class_indices[class_id]
        chosen = rng.choice(indices, size=min(500, len(indices)), replace=False)
        for index in chosen:
            image, _ = train[int(index)]
            brightness[class_id].append(float(((image + 1) / 2).mean()))
        axes[1].hist(
            brightness[class_id],
            bins=30,
            alpha=0.55,
            label="No Finding" if class_id == 0 else "Pleural Effusion",
        )
    axes[1].set_title("Mean image intensity (train)")
    axes[1].set_xlabel("Intensity in [0,1]")
    axes[1].legend()
    figure.tight_layout()
    distribution_path = output_dir / "data_distribution.png"
    figure.savefig(distribution_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    marker(logger, "DATA_VIS_READY", examples=grid_path, distribution=distribution_path)


if __name__ == "__main__":
    main()
