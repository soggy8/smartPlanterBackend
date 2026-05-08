#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_PATH="${1:-$BACKEND_DIR/datasets/plant_leaves/images/val/00169de30b63a90b8f76.jpg}"
MOISTURE="${MOISTURE:-25}"
TEMPERATURE="${TEMPERATURE:-28}"
LIGHT="${LIGHT:-600}"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image not found: $IMAGE_PATH" >&2
  echo "Usage: $0 /absolute/path/to/image.jpg" >&2
  exit 1
fi

echo "POST /sensor-data"
curl -sS -X POST "$BASE_URL/sensor-data" \
  -H "Content-Type: application/json" \
  -d "{\"moisture\":$MOISTURE,\"temperature\":$TEMPERATURE,\"light\":$LIGHT}"
echo
echo

echo "POST /image"
curl -sS -X POST "$BASE_URL/image" \
  -F "file=@$IMAGE_PATH"
echo
echo

echo "GET /analysis"
curl -sS "$BASE_URL/analysis"
echo
