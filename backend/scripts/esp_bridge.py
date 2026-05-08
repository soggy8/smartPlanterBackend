#!/usr/bin/env python3
"""Bridge ESP serial sensor output to the backend /sensor-data API."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import serial

AVG_SOIL_RE = re.compile(r"Avg Soil:\s*(\d+)\s*%?", re.IGNORECASE)
TEMP_RE = re.compile(r"Temp:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
LIGHT_PERCENT_RE = re.compile(r"Light:\s*(\d+)\s*%?", re.IGNORECASE)


@dataclass
class SensorFrame:
    moisture: int | None = None
    temperature: float | None = None
    light_percent: int | None = None

    def complete(self) -> bool:
        return self.moisture is not None and self.temperature is not None and self.light_percent is not None

    def to_api_payload(self, light_scale: int) -> dict[str, float]:
        # Backend currently expects larger light values than 0-100 percentage.
        # Scale percent to 0-1000 by default.
        light_value = int(self.light_percent or 0) * light_scale
        return {
            "moisture": int(self.moisture or 0),
            "temperature": float(self.temperature or 0.0),
            "light": light_value,
        }


def post_sensor(base_url: str, payload: dict[str, float], timeout_s: float) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/sensor-data",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read ESP serial lines and forward sensor data to API.")
    parser.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0, /dev/ttyACM0, COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--url", required=True, help="API base URL, e.g. https://<tunnel>.trycloudflare.com")
    parser.add_argument(
        "--light-scale",
        type=int,
        default=10,
        help="Multiply ESP light percent by this factor for API payload (default: 10)",
    )
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout seconds (default: 8)")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=15.0,
        help="Minimum seconds between POSTs to API (default: 15)",
    )
    args = parser.parse_args()

    frame = SensorFrame()
    last_post_ts = 0.0
    print(f"[bridge] Listening on {args.port} @ {args.baud} and posting to {args.url}")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                if not line:
                    continue

                print(f"[serial] {line}")

                m = AVG_SOIL_RE.search(line)
                if m:
                    frame.moisture = int(m.group(1))
                m = TEMP_RE.search(line)
                if m:
                    frame.temperature = float(m.group(1))
                m = LIGHT_PERCENT_RE.search(line)
                if m:
                    frame.light_percent = int(m.group(1))

                if frame.complete():
                    now = time.time()
                    elapsed = now - last_post_ts
                    if elapsed < args.min_interval:
                        wait_s = args.min_interval - elapsed
                        print(f"[post] waiting {wait_s:.1f}s to respect --min-interval")
                        time.sleep(wait_s)

                    payload = frame.to_api_payload(light_scale=args.light_scale)
                    try:
                        status, body = post_sensor(args.url, payload, timeout_s=args.timeout)
                        print(f"[post] {status} {payload} -> {body}")
                        last_post_ts = time.time()
                    except urllib.error.HTTPError as exc:
                        body = exc.read().decode("utf-8", errors="replace")
                        print(f"[post] HTTP {exc.code}: {body}", file=sys.stderr)
                    except Exception as exc:  # pragma: no cover - network/runtime
                        print(f"[post] error: {exc}", file=sys.stderr)

                    frame = SensorFrame()
    except serial.SerialException as exc:
        raise SystemExit(f"Could not open serial port {args.port}: {exc}") from exc
    except KeyboardInterrupt:
        print("\n[bridge] Stopped by user.")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
