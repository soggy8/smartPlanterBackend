#!/usr/bin/env python3
"""
Convert LaboroTomato COCO annotations into a YOLO dataset for fruit maturity.

Input assumptions:
- A dataset root containing COCO json files and images.
- JSON includes classes like:
    b_fully_ripened, b_half_ripened, b_green,
    l_fully_ripened, l_half_ripened, l_green

Output:
- datasets/tomato_fruit/images/{train,val}
- datasets/tomato_fruit/labels/{train,val}
- labels remapped to 3 classes:
    0 fruit_fully_ripened
    1 fruit_half_ripened
    2 fruit_green
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

CATEGORY_MAP = {
    "b_fully_ripened": 0,
    "l_fully_ripened": 0,
    "b_half_ripened": 1,
    "l_half_ripened": 1,
    "b_green": 2,
    "l_green": 2,
}

COCO_NAMES = ("instances_train.json", "instances_val.json", "train.json", "val.json", "instances_default.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clear_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for child in d.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def _find_json(source: Path, preferred_name: str | None, fallback_names: tuple[str, ...] = ()) -> Path:
    if preferred_name:
        p = source / preferred_name
        if p.is_file():
            return p
    for name in fallback_names or COCO_NAMES:
        p = source / name
        if p.is_file():
            return p
    matches = sorted(source.rglob("*.json"))
    if not matches:
        raise SystemExit(f"No JSON files found under {source}")
    return matches[0]


def _normalize_bbox(x: float, y: float, w: float, h: float, width: float, height: float) -> tuple[float, float, float, float]:
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return cx, cy, nw, nh


def _write_split(
    coco: dict,
    json_path: Path,
    split: str,
    out_root: Path,
    copy_images: bool,
) -> int:
    split_hint = json_path.stem
    if split_hint.startswith("instances_"):
        split_hint = split_hint.replace("instances_", "", 1)
    if split_hint == "val":
        split_hint = "test"

    images_by_id = {int(i["id"]): i for i in coco.get("images", [])}
    categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    _clear_dir(img_dir)
    _clear_dir(lbl_dir)

    n = 0
    for image_id, image in images_by_id.items():
        file_name = str(image["file_name"])
        width = float(image["width"])
        height = float(image["height"])
        src = (json_path.parent / file_name).resolve()
        if not src.is_file():
            # Common alternatives across COCO exports.
            candidates = [
                (json_path.parent / "images" / file_name).resolve(),
                (json_path.parent.parent / split / file_name).resolve(),
                (json_path.parent.parent / split_hint / file_name).resolve(),
                (json_path.parent.parent / "images" / split / file_name).resolve(),
                (json_path.parent.parent / "images" / split_hint / file_name).resolve(),
                (json_path.parent.parent / file_name).resolve(),
            ]
            src = next((c for c in candidates if c.is_file()), None)
            if src is None:
                continue

        stem = Path(file_name).stem
        dst_img = img_dir / Path(file_name).name
        dst_lbl = lbl_dir / f"{stem}.txt"

        if copy_images:
            shutil.copy2(src, dst_img)
        else:
            try:
                dst_img.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst_img)

        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            cat_name = categories.get(int(ann["category_id"]), "")
            cls = CATEGORY_MAP.get(cat_name)
            if cls is None:
                continue
            x, y, w, h = ann["bbox"]
            cx, cy, nw, nh = _normalize_bbox(float(x), float(y), float(w), float(h), width, height)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            dst_lbl.write_text("", encoding="utf-8")
        else:
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    return n


def _split_coco(coco: dict, val_ratio: float, seed: int) -> tuple[dict, dict]:
    images = list(coco.get("images", []))
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(round(len(images) * val_ratio))) if images else 0
    val_ids = {int(i["id"]) for i in images[:n_val]}

    train_coco = dict(coco)
    val_coco = dict(coco)
    train_coco["images"] = [i for i in images if int(i["id"]) not in val_ids]
    val_coco["images"] = [i for i in images if int(i["id"]) in val_ids]
    train_coco["annotations"] = [a for a in coco.get("annotations", []) if int(a["image_id"]) not in val_ids]
    val_coco["annotations"] = [a for a in coco.get("annotations", []) if int(a["image_id"]) in val_ids]
    return train_coco, val_coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LaboroTomato into YOLO fruit maturity dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Path to extracted LaboroTomato dataset root")
    parser.add_argument("--train-json", default=None, help="Train COCO json file name under source")
    parser.add_argument("--val-json", default=None, help="Val/Test COCO json file name under source")
    parser.add_argument("--single-json", default=None, help="Single COCO json to split by --val-ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio when using --single-json")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomized split")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlink")
    parser.add_argument("--out", type=Path, default=None, help="Output root (default: datasets/tomato_fruit)")
    args = parser.parse_args()

    source = args.source.resolve()
    out_root = args.out or (_repo_root() / "datasets" / "tomato_fruit")
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    if args.single_json:
        single = _find_json(source, args.single_json)
        coco = json.loads(single.read_text(encoding="utf-8"))
        train_coco, val_coco = _split_coco(coco, val_ratio=args.val_ratio, seed=args.seed)
        n_train = _write_split(train_coco, single, "train", out_root=out_root, copy_images=args.copy)
        n_val = _write_split(val_coco, single, "val", out_root=out_root, copy_images=args.copy)
    else:
        train_json = _find_json(
            source,
            args.train_json,
            fallback_names=("train.json", "instances_train.json"),
        )
        val_json = _find_json(
            source,
            args.val_json,
            fallback_names=("val.json", "test.json", "instances_val.json", "instances_test.json"),
        )
        train_coco = json.loads(train_json.read_text(encoding="utf-8"))
        val_coco = json.loads(val_json.read_text(encoding="utf-8"))
        n_train = _write_split(train_coco, train_json, "train", out_root=out_root, copy_images=args.copy)
        n_val = _write_split(val_coco, val_json, "val", out_root=out_root, copy_images=args.copy)

    print(f"Wrote fruit dataset to {out_root}")
    print(f"train images: {n_train}, val images: {n_val}")
    print("Next: python train_fruit.py --data data/tomato_fruit.yaml")


if __name__ == "__main__":
    main()
