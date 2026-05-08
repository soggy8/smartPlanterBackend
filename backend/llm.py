"""Call a local Ollama model and parse structured JSON for plant health."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


class LLMError(Exception):
    """Raised when Ollama is unreachable or the response is not valid JSON."""

    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


def _strip_code_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrapping from model output."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _build_prompt(
    sensor_summary: str,
    vision_summary: str,
    sensor_raw: dict[str, Any],
    fruit_summary: str,
) -> str:
    return f"""You are an agricultural AI assistant.

Sensor data (raw):
- Soil moisture: {sensor_raw.get("moisture")}
- Temperature (°C): {sensor_raw.get("temperature")}
- Light (lux or arbitrary units): {sensor_raw.get("light")}

Sensor interpretation:
{sensor_summary}

Vision analysis:
{vision_summary}

Fruit maturity analysis:
{fruit_summary}

Respond with JSON ONLY (no markdown fences) using exactly these keys:
"health_score" (number 0-100),
"status" (short string),
"problem" (short string),
"advice" (short actionable string),
"harvest_estimate_days" (integer, assume a tomato plant in typical home conditions).

Example shape:
{{"health_score": 70, "status": "...", "problem": "...", "advice": "...", "harvest_estimate_days": 21}}
"""


def generate_health_json(
    sensor_summary: str,
    vision_summary: str,
    sensor_raw: dict[str, Any],
    fruit_summary: str = "No fruit maturity detections available",
) -> dict[str, Any]:
    """
    Step 1: build prompt. Step 2: POST to Ollama. Step 3: parse JSON object.

    Raises:
        LLMError: on HTTP errors or invalid JSON payload.
    """
    prompt = _build_prompt(sensor_summary, vision_summary, sensor_raw, fruit_summary)
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network
        raise LLMError(f"Ollama HTTP error: {exc}") from exc

    data = response.json()
    raw_text = str(data.get("response", "")).strip()
    cleaned = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError("Model did not return valid JSON", raw_text=raw_text) from exc

    required = {"health_score", "status", "problem", "advice", "harvest_estimate_days"}
    if not required.issubset(parsed.keys()):
        raise LLMError("JSON missing required keys", raw_text=raw_text)

    def _to_int(value: Any, field: str) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise LLMError(f"Invalid numeric field {field!r}", raw_text=raw_text) from exc

    health_score = max(0, min(100, _to_int(parsed["health_score"], "health_score")))

    # Normalize types lightly for API consumers.
    return {
        "health_score": health_score,
        "status": str(parsed["status"]),
        "problem": str(parsed["problem"]),
        "advice": str(parsed["advice"]),
        "harvest_estimate_days": _to_int(parsed["harvest_estimate_days"], "harvest_estimate_days"),
    }
