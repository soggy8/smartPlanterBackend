"""Interpret fruit-stage detections for harvest-related reasoning."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

_FRUIT_CLASS_ALIASES: dict[str, str] = {
    "fruit_green": "fruit_green",
    "b_green": "fruit_green",
    "l_green": "fruit_green",
    "fruit_half_ripened": "fruit_half_ripened",
    "b_half_ripened": "fruit_half_ripened",
    "l_half_ripened": "fruit_half_ripened",
    "fruit_fully_ripened": "fruit_fully_ripened",
    "b_fully_ripened": "fruit_fully_ripened",
    "l_fully_ripened": "fruit_fully_ripened",
}


def interpret_detections(
    detections: Iterable[Mapping[str, Any]],
    conf_threshold: float = 0.25,
) -> str:
    """Summarize fruit maturity counts from detections."""
    counts: Counter[str] = Counter()
    for det in detections:
        name = str(det.get("class_name", "")).strip()
        conf = float(det.get("confidence", 0.0))
        if conf < conf_threshold:
            continue
        stage = _FRUIT_CLASS_ALIASES.get(name, name)
        counts[stage] += 1

    total = sum(counts.values())
    if total == 0:
        return "No tomato fruits confidently detected in the image"

    green = counts.get("fruit_green", 0)
    half = counts.get("fruit_half_ripened", 0)
    full = counts.get("fruit_fully_ripened", 0)

    parts = [f"Detected {total} tomatoes"]
    parts.append(f"{green} green")
    parts.append(f"{half} half-ripened")
    parts.append(f"{full} fully ripened")

    if full >= max(green, half) and full > 0:
        parts.append("harvest window appears open for some fruits")
    elif green > full + half:
        parts.append("most fruits still need time to mature")

    return "; ".join(parts)
