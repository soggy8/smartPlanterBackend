# Smart Planter

Prototype stack for an AI-assisted **smart planter**: a Python backend turns **sensor readings** and **camera images** into a structured plant-health summary (including optional fruit-ripeness hints), using YOLO-style vision models and an LLM for the final narrative output.

This repository is organized so hardware and firmware can grow next to the backend in one place.

## Repository layout

| Path | Purpose |
|------|--------|
| [`backend/`](backend/) | Python service, training scripts, tests, web UI, and YOLO dataset configs (`data/*.yaml`). **Run commands from this directory** unless noted otherwise. |
| [`stl/`](stl/) | Reserved for **3D-printable** enclosure or planter parts (`.stl` files). Empty for now; add models here when ready. |
| [`esp32/`](esp32/) | Reserved for **ESP32 firmware** (e.g. Arduino / ESP-IDF) that reads sensors and talks to the backend. Empty for now. |

### Inside `backend/`

- **`main.py`** — FastAPI app: ingest sensors and image, run inference, return analysis JSON.
- **`yolo_model.py`**, **`fruit_model.py`** — Leaf-health and fruit-maturity detectors (Ultralytics YOLO).
- **`vision.py`**, **`fruit.py`**, **`sensors.py`** — Turn raw detections and sensor numbers into short text summaries.
- **`llm.py`** — Calls a configured LLM API and expects structured JSON (health score, status, advice, etc.).
- **`train.py`**, **`train_fruit.py`** — Train detectors using `data/plant_leaves.yaml` and `data/tomato_fruit.yaml`.
- **`scripts/`** — Dataset preparation (`prepare_*`), serial/Web helpers (`esp_bridge.py`, `webcam_uploader.py`), and `test_pipeline.sh` for curl-based smoke tests.
- **`web/index.html`** — Simple browser UI served at `GET /ui`.

Prepared datasets and downloaded weights are **not** committed (see `.gitignore`). After preparing data, they live under `backend/datasets/` and `backend/weights/` by default.

## Quick start (backend)

Python 3.11+ recommended.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Demo UI: [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)

### Typical request flow

1. `POST /sensor-data` — JSON body: `moisture`, `temperature`, `light`.
2. `POST /image` — multipart file upload (e.g. JPEG from the planter camera).
3. `GET /analysis` — Combined sensor + vision + LLM response.

### Weights and demo mode

- Leaf model weights default to `weights/best.pt` (relative to the **current working directory** — keep using `cd backend`).
- Override with `YOLO_WEIGHTS`.
- If weights are missing, set `DEMO_FAKE_VISION=1` to use deterministic fake detections for UI/testing.

Fruit model is optional; if `weights/fruit_best.pt` is absent, analysis still runs without fruit detections.

### Tests

```bash
cd backend
pytest tests/ -q
```

### Optional: pipeline smoke test

With the server running:

```bash
cd backend
./scripts/test_pipeline.sh /path/to/a/plant/photo.jpg
```

## Training and data (short pointers)

- Leaf dataset prep: `backend/scripts/prepare_kaggle_tomato_dataset.py` (writes under `backend/datasets/plant_leaves/` by default).
- Fruit dataset prep: `backend/scripts/prepare_laboro_tomato.py`.
- Then train from `backend/`:

```bash
cd backend
python train.py --data data/plant_leaves.yaml
python train_fruit.py --data data/tomato_fruit.yaml
```

## Environment variables (high level)

| Variable | Role |
|----------|------|
| `YOLO_WEIGHTS`, `YOLO_CONF` | Leaf detector checkpoint and confidence threshold. |
| `FRUIT_YOLO_WEIGHTS`, `FRUIT_YOLO_CONF` | Fruit detector (optional). |
| `DEMO_FAKE_VISION` | Use fake leaf detections when weights are absent. |
| `ANALYSIS_INCLUDE_DEBUG` | Set to `0` to omit debug block in `/analysis`. |

LLM-related settings are read inside `llm.py` (API URL, key, model id — use `.env` locally and do not commit secrets).

## Moving old local folders

If you previously kept `datasets/` or `weights/` at the **repository root**, move them into `backend/datasets/` and `backend/weights/` so paths match the YAML configs and training scripts.

## License

Add a license file if you plan to publish this repo publicly.
