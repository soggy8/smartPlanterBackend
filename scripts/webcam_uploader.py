#!/usr/bin/env python3
"""Capture webcam frames on an interval and upload them to the API."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time

import httpx

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV is required for webcam capture. Install with:\n"
        "  pip install opencv-python\n"
        "(Use this on your laptop environment, not necessarily on the cloud server.)"
    ) from exc


def capture_jpeg(device_index: int, width: int | None, height: int | None, jpeg_quality: int) -> bytes:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam device index {device_index}")
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from webcam")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("Failed to encode webcam frame to JPEG")
        return encoded.tobytes()
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload webcam images to /image periodically.")
    parser.add_argument("--url", required=True, help="API base URL (cloudflare tunnel or localhost)")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between captures (default: 60)")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)")
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="After each upload, call GET /analysis and print result snippet",
    )
    parser.add_argument(
        "--save-dir",
        type=pathlib.Path,
        default=None,
        help="Optional local folder to save captured JPEGs for debugging",
    )
    args = parser.parse_args()

    if args.interval < 3:
        raise SystemExit("--interval should be >= 3 seconds to avoid API overload")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in range 1..100")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    base = args.url.rstrip("/")
    print(f"[webcam] Uploading to {base}/image every {args.interval:.1f}s")
    print(f"[webcam] device={args.device}, run_analysis={args.run_analysis}")

    with httpx.Client(timeout=30.0) as client:
        while True:
            t0 = time.time()
            try:
                jpeg = capture_jpeg(args.device, args.width, args.height, args.jpeg_quality)
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_{ts}.jpg"

                if args.save_dir:
                    (args.save_dir / filename).write_bytes(jpeg)

                files = {"file": (filename, jpeg, "image/jpeg")}
                up = client.post(f"{base}/image", files=files)
                up.raise_for_status()
                print(f"[upload] ok {up.status_code} {filename}")

                if args.run_analysis:
                    ar = client.get(f"{base}/analysis")
                    ar.raise_for_status()
                    data = ar.json()
                    summary = {
                        "health_score": data.get("health_score"),
                        "status": data.get("status"),
                        "harvest_estimate_days": data.get("harvest_estimate_days"),
                    }
                    print(f"[analysis] {json.dumps(summary)}")

            except Exception as exc:
                print(f"[error] {exc}")

            elapsed = time.time() - t0
            sleep_s = max(0.0, args.interval - elapsed)
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()

