"""Run YOLO fruit-maturity inference from uploaded image bytes."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

from PIL import Image

FRUIT_CLASS_NAMES: tuple[str, ...] = (
    "fruit_fully_ripened",
    "fruit_half_ripened",
    "fruit_green",
)

_model = None
_model_path: str | None = None


def _weights_path() -> str:
    return os.environ.get("FRUIT_YOLO_WEIGHTS", "weights/fruit_best.pt")


def get_model():
    """Lazy-load YOLO fruit model."""
    global _model, _model_path
    path = _weights_path()
    if _model is not None and _model_path == path:
        return _model

    from ultralytics import YOLO  # local import

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fruit YOLO weights not found at {path!r}. "
            "Train with train_fruit.py and set FRUIT_YOLO_WEIGHTS."
        )

    _model = YOLO(path)
    _model_path = path
    return _model


def run_inference(image_bytes: bytes, conf: float | None = None) -> list[dict[str, Any]]:
    """Run fruit detector and return class/confidence/bbox detections."""
    if conf is None:
        conf = float(os.environ.get("FRUIT_YOLO_CONF", "0.25"))

    model = get_model()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=image, conf=conf, verbose=False)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    names = result.names or {}
    output: list[dict[str, Any]] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        class_name = str(names.get(cls_id, FRUIT_CLASS_NAMES[cls_id] if cls_id < len(FRUIT_CLASS_NAMES) else cls_id))
        output.append(
            {
                "class_name": class_name,
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist(),
            }
        )
    return output
