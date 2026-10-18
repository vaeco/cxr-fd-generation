#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, marker, save_json, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patient-disjoint CheXpert manifests")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disease", default="Pleural Effusion")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-per-class", type=int, default=12000)
    parser.add_argument("--max-val-per-class", type=int, default=2000)
    parser.add_argument("--max-test-per-class", type=int, default=2000)
    parser.add_argument("--min-per-class", type=int, default=100)
    return parser.parse_args()


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0] == data_root.name:
        parts = parts[1:]
    return (data_root / Path(*parts)).resolve()


def patient_from_path(raw_path: str) -> str:
    match = re.search(r"patient\d+", str(raw_path))
    if match:
        return match.group(0)
    # Deterministic fallback for an unexpected path layout.
    return str(Path(str(raw_path)).parent)


def stratified_patient_split(frame: pd.DataFrame, seed: int) -> dict[str, str]:
    patient_table = (
        frame.groupby("patient_id", as_index=False)["class_id"].max().rename(
            columns={"class_id": "has_disease"}
        )
    )
    train_patients, holdout_patients = train_test_split(
        patient_table,
        test_size=0.20,
        random_state=seed,
        stratify=patient_table["has_disease"],
    )
    val_patients, test_patients = train_test_split(
        holdout_patients,
        test_size=0.50,
        random_state=seed + 1,
        stratify=holdout_patients["has_disease"],
    )
    mapping: dict[str, str] = {}
    mapping.update({patient: "train" for patient in train_patients["patient_id"]})
    mapping.update({patient: "val" for patient in val_patients["patient_id"]})
    mapping.update({patient: "test" for patient in test_patients["patient_id"]})
    return mapping


def balance_split(
    frame: pd.DataFrame,
    split: str,
    cap: int,
    min_per_class: int,
    seed: int,
) -> pd.DataFrame:
    subset = frame[frame["split"] == split]
    counts = subset.groupby("class_id").size().to_dict()
    if set(counts) != {0, 1}:
        raise RuntimeError(f"split={split} does not contain both classes: {counts}")
    take = min(int(counts[0]), int(counts[1]), int(cap))
    if take < min_per_class:
        raise RuntimeError(
            f"split={split} has only {take} usable images/class; need {min_per_class}"
        )
    pieces = []
    for class_id in (0, 1):
        class_rows = subset[subset["class_id"] == class_id]
        pieces.append(class_rows.sample(n=take, random_state=seed + class_id))
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("manifest", output_dir / "prepare_manifest.log")
    csv_path = args.data_root / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    logger.info("Reading %s", csv_path)
    source = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", "No Finding", args.disease}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(f"CheXpert CSV missing columns: {sorted(missing_columns)}")

    source = source[source["Frontal/Lateral"] == "Frontal"].copy()
    disease_label = source[args.disease].fillna(0)
    # Exclude uncertain (-1) disease labels from both binary classes.
    normal = source[(source["No Finding"] == 1) & (disease_label == 0)].copy()
    disease = source[(disease_label == 1) & (source["No Finding"] != 1)].copy()
    normal["class_id"] = 0
    normal["class_name"] = "No Finding"
    disease["class_id"] = 1
    disease["class_name"] = args.disease
    selected = pd.concat([normal, disease], ignore_index=True)
    selected["patient_id"] = selected["Path"].map(patient_from_path)
    selected["path"] = selected["Path"].map(
        lambda value: str(resolve_image_path(value, args.data_root))
    )
    exists = selected["path"].map(lambda value: Path(value).is_file())
    missing_count = int((~exists).sum())
    selected = selected[exists].copy().reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No usable image files were resolved; check --data-root")
    marker(
        logger,
        "FILTER",
        frontal=len(source),
        usable=len(selected),
        missing_files=missing_count,
        patients=selected["patient_id"].nunique(),
    )

    mapping = stratified_patient_split(selected, args.seed)
    selected["split"] = selected["patient_id"].map(mapping)
    caps = {
        "train": args.max_train_per_class,
        "val": args.max_val_per_class,
        "test": args.max_test_per_class,
    }
    balanced = pd.concat(
        [
            balance_split(
                selected,
                split=split,
                cap=cap,
                min_per_class=args.min_per_class,
                seed=args.seed + index * 10,
            )
            for index, (split, cap) in enumerate(caps.items())
        ],
        ignore_index=True,
    )
    columns = ["path", "class_id", "class_name", "patient_id", "split"]
    manifest_path = output_dir / "manifest.csv"
    balanced[columns].to_csv(manifest_path, index=False)

    patient_sets = {
        split: set(balanced.loc[balanced["split"] == split, "patient_id"])
        for split in caps
    }
    overlap = {
        "train_val": len(patient_sets["train"] & patient_sets["val"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "val_test": len(patient_sets["val"] & patient_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Patient leakage detected: {overlap}")
    count_table = (
        balanced.groupby(["split", "class_id", "class_name"]).size().rename("images")
    )
    logger.info("\n%s", count_table.to_string())
    summary = {
        "data_root": str(args.data_root.resolve()),
        "disease": args.disease,
        "seed": args.seed,
        "missing_files": missing_count,
        "rows": int(len(balanced)),
        "counts": {
            split: {
                str(class_id): int(
                    len(
                        balanced[
                            (balanced["split"] == split)
                            & (balanced["class_id"] == class_id)
                        ]
                    )
                )
                for class_id in (0, 1)
            }
            for split in caps
        },
        "patients": {split: len(values) for split, values in patient_sets.items()},
        "patient_overlap": overlap,
    }
    save_json(summary, output_dir / "summary.json")
    marker(logger, "LEAKAGE_CHECK", **overlap)
    marker(logger, "MANIFEST_READY", path=manifest_path, rows=len(balanced))


if __name__ == "__main__":
    main()
