"""Load a YOLOv8 model once and run inference on in-memory image bytes."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

from PIL import Image

# Expected class names for the fine-tuned plant-health detector.
PLANT_CLASS_NAMES: tuple[str, ...] = (
    "healthy_leaf",
    "yellow_leaf",
    "spotted_leaf",
    "damaged_leaf",
)

_model = None
_model_path: str | None = None


def _weights_path() -> str:
    return os.environ.get("YOLO_WEIGHTS", "weights/best.pt")


def _demo_fake_enabled() -> bool:
    return os.environ.get("DEMO_FAKE_VISION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _fake_detections() -> list[dict[str, Any]]:
    """Deterministic demo detections when no real weights are available."""
    return [
        {"class_name": "yellow_leaf", "confidence": 0.78, "bbox": None},
        {"class_name": "healthy_leaf", "confidence": 0.62, "bbox": None},
    ]


def get_model():
    """Lazy-load Ultralytics YOLO so import stays light for tests without GPU."""
    global _model, _model_path
    path = _weights_path()
    if _model is not None and _model_path == path:
        return _model

    from ultralytics import YOLO  # local import

    if not os.path.isfile(path):
        if _demo_fake_enabled():
            _model = None
            _model_path = None
            return None
        raise FileNotFoundError(
            f"YOLO weights not found at {path!r}. "
            "Train with train.py and set YOLO_WEIGHTS, "
            "or set DEMO_FAKE_VISION=1 for demo detections."
        )

    _model = YOLO(path)
    _model_path = path
    return _model


def run_inference(image_bytes: bytes, conf: float | None = None) -> list[dict[str, Any]]:
    """
    Run YOLO on ``image_bytes`` and return a list of detection dicts.

    Each dict: ``class_name``, ``confidence``, ``bbox`` (xyxy list or None).
    """
    if conf is None:
        conf = float(os.environ.get("YOLO_CONF", "0.25"))

    # Demo path: missing weights but explicit demo flag.
    if not os.path.isfile(_weights_path()) and _demo_fake_enabled():
        return _fake_detections()

    model = get_model()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    output: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return output

    names = result.names or {}
    for box in result.boxes:
        cls_id = int(box.cls.item())
        class_name = str(names.get(cls_id, PLANT_CLASS_NAMES[cls_id] if cls_id < len(PLANT_CLASS_NAMES) else cls_id))
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        output.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": xyxy,
            }
        )

    return output
