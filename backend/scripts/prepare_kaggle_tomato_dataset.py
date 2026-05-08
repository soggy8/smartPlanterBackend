#!/usr/bin/env python3
"""
Prepare Plant Village (Kaggle) tomato leaves for YOLOv8 training in this repo.

The Kaggle release is image-classification style (one folder per disease). YOLO
training here uses one full-frame box per image with a mapped class id:

  0 healthy_leaf
  1 yellow_leaf
  2 spotted_leaf
  3 damaged_leaf

Default source: kagglehub dataset ``charuchaudhry/plantvillage-tomato-leaf-dataset``
(``plantvillage/`` inside the extracted version folder).

Outputs (under ``backend/`` by default):

  datasets/plant_leaves/images/{train,val}/*.jpg
  datasets/plant_leaves/labels/{train,val}/*.txt   # one YOLO line: cls 0.5 0.5 1 1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

# Plant Village folder name -> YOLO class id (see data/plant_leaves.yaml).
FOLDER_TO_CLASS_ID: dict[str, int] = {
    "Tomato___healthy": 0,
    # Strong yellowing / chlorosis phenotype in this dataset family.
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 1,
    # Spot / blight / mold diseases.
    "Tomato___Bacterial_spot": 2,
    "Tomato___Early_blight": 2,
    "Tomato___Late_blight": 2,
    "Tomato___Leaf_Mold": 2,
    "Tomato___Septoria_leaf_spot": 2,
    "Tomato___Target_Spot": 2,
    # Mite feeding damage + mosaic distortion.
    "Tomato___Spider_mites Two-spotted_spider_mite": 3,
    "Tomato___Tomato_mosaic_virus": 3,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_kaggle_plantvillage_dir() -> Path:
    import kagglehub

    base = Path(kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset"))
    return base / "plantvillage"


def _collect_images(source: Path) -> dict[int, list[Path]]:
    by_class: dict[int, list[Path]] = defaultdict(list)
    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == "plantvillage":
            # Skip nested duplicate tree if present.
            continue
        cls_id = FOLDER_TO_CLASS_ID.get(folder.name)
        if cls_id is None:
            continue
        for img in folder.rglob("*"):
            if not img.is_file():
                continue
            if img.suffix not in IMAGE_SUFFIXES:
                continue
            by_class[cls_id].append(img)
    return by_class


def _unique_stem(src: Path) -> str:
    h = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:20]
    ext = src.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return f"{h}{ext}"


def _write_yolo_full_image_label(label_path: Path, class_id: int) -> None:
    # Normalized cx, cy, w, h covering the full image.
    label_path.write_text(f"{class_id} 0.5 0.5 1 1\n", encoding="utf-8")


def _clear_dir(d: Path) -> None:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return
    for child in d.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Plant Village tomato dataset for YOLOv8.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to .../plantvillage (folder containing Tomato___*). "
        "Default: download/locate via kagglehub.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dataset root (default: <repo>/datasets/plant_leaves)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction per class for validation")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for split")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (slower, uses more disk)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    source = args.source or _default_kaggle_plantvillage_dir()
    out_root = args.out or (repo / "datasets" / "plant_leaves")

    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    by_class = _collect_images(source)
    if not by_class:
        raise SystemExit(f"No images found under {source}. Expected Tomato___* class folders.")

    rng = random.Random(args.seed)

    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        _clear_dir(d)

    train_n = val_n = 0
    for cls_id, paths in sorted(by_class.items(), key=lambda kv: kv[0]):
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        n_val = int(round(n * args.val_ratio))
        n_val = max(0, min(n, n_val))
        if n_val == 0 and n > 1:
            n_val = 1
        val_paths = set(paths[:n_val])
        for src in paths:
            stem = _unique_stem(src)
            is_val = src in val_paths
            dst_img_dir = img_val if is_val else img_train
            dst_lbl_dir = lbl_val if is_val else lbl_train
            dst_img = dst_img_dir / stem
            dst_lbl = dst_lbl_dir / (Path(stem).stem + ".txt")

            if args.copy:
                shutil.copy2(src, dst_img)
            else:
                try:
                    dst_img.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst_img)

            _write_yolo_full_image_label(dst_lbl, cls_id)
            if is_val:
                val_n += 1
            else:
                train_n += 1

        print(f"class {cls_id}: total={n}, val={n_val}, train={n - n_val}")

    print(f"\nWrote train={train_n}, val={val_n} samples under {out_root}")
    print("Next: python train.py --data data/plant_leaves.yaml")


if __name__ == "__main__":
    main()
