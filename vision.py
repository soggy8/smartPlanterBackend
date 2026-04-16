"""Turn YOLO detection labels into short, human-readable vision summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

# Class id -> phrase (aligned with training data/plant_leaves.yaml order).
_CLASS_MESSAGES: dict[str, str] = {
    "healthy_leaf": "healthy-looking foliage",
    "yellow_leaf": "leaf discoloration detected",
    "spotted_leaf": "possible disease spots",
    "damaged_leaf": "physical damage or pests",
}


def interpret_detections(
    detections: Iterable[Mapping[str, Any]],
    conf_threshold: float = 0.25,
) -> str:
    """
    Map each detection class to a phrase and merge into one summary.

    Detections are dicts with at least ``class_name`` and ``confidence``.
    """
    counts: Counter[str] = Counter()
    for det in detections:
        name = str(det.get("class_name", "")).strip()
        conf = float(det.get("confidence", 0.0))
        if not name or conf < conf_threshold:
            continue
        counts[name] += 1

    if not counts:
        return "No clear leaf issues detected in the image"

    stress_classes = {"yellow_leaf", "spotted_leaf", "damaged_leaf"}
    present_stress = [c for c in counts if c in stress_classes]

    phrases: list[str] = []
    for cls, _n in counts.most_common():
        if cls in stress_classes:
            phrases.append(_CLASS_MESSAGES.get(cls, cls))
        elif cls == "healthy_leaf":
            # Mention healthy leaves only when no obvious stress signals.
            if not present_stress:
                phrases.append(_CLASS_MESSAGES["healthy_leaf"])

    if not phrases and "healthy_leaf" in counts:
        phrases.append(_CLASS_MESSAGES["healthy_leaf"])

    # De-duplicate while keeping order.
    seen: set[str] = set()
    ordered: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    if not ordered:
        return "No clear leaf issues detected in the image"

    if len(ordered) == 1:
        base = f"Leaves show {ordered[0]}"
    else:
        joined = ", ".join(ordered[:-1]) + f" and {ordered[-1]}"
        base = f"Leaves show {joined}"

    return base[0].upper() + base[1:]
