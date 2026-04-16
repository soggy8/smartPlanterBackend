"""Train a YOLOv8n detector on the custom leaf-health dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on plant leaf classes.")
    parser.add_argument(
        "--data",
        default="data/plant_leaves.yaml",
        help="Path to dataset yaml (default: data/plant_leaves.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Starting checkpoint (default: pretrained yolov8n.pt)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
