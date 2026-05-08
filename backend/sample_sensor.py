#!/usr/bin/env python3
"""Generate sample sensor JSON for testing without hardware."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request


def random_sensor() -> dict[str, int]:
    return {
        "moisture": random.randint(5, 95),
        "temperature": random.randint(18, 36),
        "light": random.randint(50, 1200),
    }


def post_sensor(base_url: str, payload: dict[str, int]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/sensor-data",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or POST random sensor samples.")
    parser.add_argument(
        "--post",
        action="store_true",
        help="POST JSON to /sensor-data on the API (default URL http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="API base URL when using --post",
    )
    args = parser.parse_args()

    payload = random_sensor()
    if args.post:
        post_sensor(args.url, payload)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
