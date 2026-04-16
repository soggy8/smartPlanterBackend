"""Convert raw sensor readings into human-readable environment states."""

from __future__ import annotations

from typing import Any, Mapping


def interpret_sensor_readings(data: Mapping[str, Any]) -> str:
    """
    Apply rule-based thresholds and return one summary sentence.

    Rules:
    - moisture < 30 -> dry; > 80 -> overwatered; else optimal
    - temperature > 30 -> high temperature stress
    - light < 300 -> low light
    """
    moisture = float(data["moisture"])
    temperature = float(data["temperature"])
    light = float(data["light"])

    parts: list[str] = []

    if moisture < 30:
        parts.append("dry soil moisture")
    elif moisture > 80:
        parts.append("overwatered soil moisture")
    else:
        parts.append("optimal soil moisture")

    if temperature > 30:
        parts.append("high temperature stress")

    if light < 300:
        parts.append("low light")

    summary = "; ".join(parts)
    return summary[0].upper() + summary[1:] if summary else summary
