"""FastAPI entrypoint for the AI plant health monitoring prototype."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

import llm
import sensors
import vision
import yolo_model

app = FastAPI(title="AI Plant Health Monitoring", version="0.1.0")


@app.get("/")
def root() -> dict[str, Any]:
    """Browser-friendly entry: lists main routes (``/`` has no HTML UI)."""
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "flow": "POST /sensor-data → POST /image → GET /analysis",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Browsers request this automatically; avoid noisy 404s in logs."""
    return Response(status_code=204)


class SensorPayload(BaseModel):
    moisture: float
    temperature: float
    light: float


class VisionDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float] | None = None


class DebugPayload(BaseModel):
    vision_detections: list[VisionDetection]
    vision_summary: str
    sensor_summary: str
    sensor_raw: dict[str, Any]


class AnalysisResponse(BaseModel):
    health_score: int = Field(ge=0, le=100)
    status: str
    problem: str
    advice: str
    harvest_estimate_days: int
    debug: DebugPayload | None = None


# --- In-memory demo state (single concurrent user) ---
_last_sensor: dict[str, Any] | None = None
_last_image_bytes: bytes | None = None


def _include_debug() -> bool:
    return os.environ.get("ANALYSIS_INCLUDE_DEBUG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


@app.post("/sensor-data")
def post_sensor_data(payload: SensorPayload) -> dict[str, str]:
    """Step 0a: accept JSON sensor readings and store them in memory."""
    global _last_sensor
    _last_sensor = payload.model_dump()
    return {"status": "ok", "message": "sensor data stored"}


@app.post("/image")
async def post_image(file: UploadFile = File(...)) -> dict[str, str]:
    """Step 0b: accept an uploaded plant image and store bytes in memory."""
    global _last_image_bytes
    _last_image_bytes = await file.read()
    if not _last_image_bytes:
        raise HTTPException(status_code=400, detail="empty upload")
    return {"status": "ok", "message": "image stored"}


@app.get("/analysis", response_model=AnalysisResponse)
def get_analysis() -> AnalysisResponse:
    """
    Full pipeline:
    1) ensure sensor + image exist
    2) YOLO inference
    3) vision + sensor interpretation
    4) LLM structured JSON
    """
    if _last_sensor is None:
        raise HTTPException(status_code=400, detail="no sensor data yet; POST /sensor-data first")
    if _last_image_bytes is None:
        raise HTTPException(status_code=400, detail="no image yet; POST /image first")

    # Step 2: vision model
    try:
        detections = yolo_model.run_inference(_last_image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - ultralytics runtime
        raise HTTPException(status_code=500, detail=f"vision inference failed: {exc}") from exc

    conf_threshold = float(os.environ.get("YOLO_CONF", "0.25"))
    # Step 3a: phrase-level vision summary
    vision_summary = vision.interpret_detections(detections, conf_threshold=conf_threshold)
    # Step 3b: sensor summary
    sensor_summary = sensors.interpret_sensor_readings(_last_sensor)

    # Step 4: LLM JSON
    try:
        llm_payload = llm.generate_health_json(sensor_summary, vision_summary, _last_sensor)
    except llm.LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail={"message": str(exc), "raw": exc.raw_text},
        ) from exc

    debug = None
    if _include_debug():
        debug = DebugPayload(
            vision_detections=[VisionDetection(**d) for d in detections],
            vision_summary=vision_summary,
            sensor_summary=sensor_summary,
            sensor_raw=dict(_last_sensor),
        )

    return AnalysisResponse(
        health_score=llm_payload["health_score"],
        status=llm_payload["status"],
        problem=llm_payload["problem"],
        advice=llm_payload["advice"],
        harvest_estimate_days=llm_payload["harvest_estimate_days"],
        debug=debug,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
