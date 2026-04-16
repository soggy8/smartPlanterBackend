# AI Plant Health Monitoring (prototype)

FastAPI service that combines **YOLOv8** leaf detections, **rule-based sensor interpretation**, and a **local Ollama** LLM to return structured plant-health advice.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start [Ollama](https://ollama.com/), then pull a model (match `OLLAMA_MODEL`):

```bash
ollama pull llama3.2
```

### Weights and demo mode

- **Real inference**: train (see below) and set `YOLO_WEIGHTS` to your `best.pt`, or copy it to `weights/best.pt` (default path).
- **No weights yet**: run with demo detections:

```bash
export DEMO_FAKE_VISION=1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Run the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive OpenAPI.

## Environment variables


| Variable                 | Default                  | Meaning                                                            |
| ------------------------ | ------------------------ | ------------------------------------------------------------------ |
| `YOLO_WEIGHTS`           | `weights/best.pt`        | Path to trained `.pt` weights                                      |
| `YOLO_CONF`              | `0.25`                   | Minimum confidence for vision phrasing                             |
| `DEMO_FAKE_VISION`       | unset                    | Set to `1` / `true` to use fake YOLO output if weights are missing |
| `OLLAMA_HOST`            | `http://127.0.0.1:11434` | Ollama server URL                                                  |
| `OLLAMA_MODEL`           | `llama3.2`               | Model name known to Ollama                                         |
| `ANALYSIS_INCLUDE_DEBUG` | `1`                      | Set to `0` to hide the `debug` block on `/analysis`                |


## Example `curl` flow

Sensor JSON (store in memory):

```bash
curl -sS -X POST http://127.0.0.1:8000/sensor-data \
  -H 'Content-Type: application/json' \
  -d '{"moisture":25,"temperature":28,"light":600}'
```

Upload an image (`file` field):

```bash
curl -sS -X POST http://127.0.0.1:8000/image \
  -F 'file=@/path/to/plant.jpg'
```

Run the full pipeline:

```bash
curl -sS http://127.0.0.1:8000/analysis
```

Order matters: **sensor** and **image** must be posted before **GET /analysis**.

## Sample sensor generator

Print random readings:

```bash
python sample_sensor.py
```

Post them to the running API:

```bash
python sample_sensor.py --post --url http://127.0.0.1:8000
```

## Dataset layout for training

Paths in `[data/plant_leaves.yaml](data/plant_leaves.yaml)` point to `datasets/plant_leaves/`:

```
datasets/plant_leaves/
  images/train/*.jpg   (or .png)
  images/val/*.jpg
  labels/train/*.txt   # YOLO format, same stem as image
  labels/val/*.txt
```

Class indices (must match yaml):


| id  | name         |
| --- | ------------ |
| 0   | healthy_leaf |
| 1   | yellow_leaf  |
| 2   | spotted_leaf |
| 3   | damaged_leaf |


## Kaggle: Plant Village tomato leaves (`charuchaudhry/plantvillage-tomato-leaf-dataset`)

This repo includes a helper that **downloads** (via `kagglehub`) and **converts** the classification-style folders into YOLO layout under `datasets/plant_leaves/`.

1. Install the downloader (once):

```bash
pip install -r requirements.txt
```

2. Build `datasets/plant_leaves/` (default: **symlinks** into the kagglehub cache to save disk; add `--copy` to duplicate files instead):

```bash
python scripts/prepare_kaggle_tomato_dataset.py
```

That calls `kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")`, then maps each `Tomato___…` folder to one of the four YOLO classes and writes one full-image label per file (`class_id 0.5 0.5 1 1`). Mapping is heuristic but demo-friendly:

- `Tomato___healthy` → `healthy_leaf`
- `Tomato___Tomato_Yellow_Leaf_Curl_Virus` → `yellow_leaf`
- Bacterial / blight / mold / spot diseases → `spotted_leaf`
- Spider mites + mosaic virus → `damaged_leaf`

Use an explicit cache path if you already downloaded:

```bash
python scripts/prepare_kaggle_tomato_dataset.py \
  --source ~/.cache/kagglehub/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset/versions/1/plantvillage
```

Re-run the script anytime to **reset** `images/*` and `labels/*` and rebuild the split (`--val-ratio`, `--seed`).

Symlinks point into the kagglehub cache; if you delete that cache, rerun `prepare_kaggle_tomato_dataset.py` or use `--copy` for a self-contained copy under `datasets/plant_leaves/`.

## Train YOLOv8n

From the project root (GPU recommended):

```bash
python train.py --data data/plant_leaves.yaml --epochs 40 --imgsz 640 --weights yolov8n.pt
```

Ultralytics writes weights under `runs/detect/train*/weights/best.pt` (incrementing `train`, `train2`, …). Point inference at the newest run:

```bash
export YOLO_WEIGHTS=runs/detect/train/weights/best.pt
```

## Tests

```bash
pytest tests/ -q
```

## Thunder Compute (A6000) — first-time GPU training

Use the cloud GPU mainly for **training**; run the API locally if you prefer.

Official docs: [Thunder Compute quickstart](https://www.thundercompute.com/docs/cli/quickstart), [creating instances](https://www.thundercompute.com/docs/console/operations/creating-instances), [SSH guide](https://www.thundercompute.com/docs/guides/ssh-on-thunder-compute), [connecting to instances](https://www.thundercompute.com/docs/cli/operations/connecting-to-instances).

1. **Account + CLI** — Sign up, install the `tnr` CLI, and verify it runs (`tnr` help / status per docs).
2. **SSH key** — Generate if needed: `ssh-keygen -t ed25519 -C "you@example.com"`. Add the **public** key to Thunder (`tnr ssh-keys add`, confirm with `tnr ssh-keys list`).
3. **Create instance** — Pick **A6000** in the console or CLI and attach your SSH key (default user is often `ubuntu`).
4. **Connect** — Prefer `tnr connect …` per Thunder’s docs. For manual SSH, use the **host**, **user**, and **port** from `tnr status` (port is often not 22):
  ```bash
   ssh -i ~/.ssh/id_ed25519 -p <port> ubuntu@<instance-ip>
  ```
5. **Check GPU** — On the VM: `nvidia-smi` should list the **A6000**.
6. **Project on VM** — `git clone` your repo or `rsync`/`scp` the folder. Create a venv and `pip install -r requirements.txt`.
7. **CUDA check** (if training fails to use GPU):
  ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
8. **Train** — Upload your dataset into `datasets/plant_leaves/...`, then `python train.py` from the project root. Use the produced `best.pt` with `YOLO_WEIGHTS`.
9. **Copy weights home** (optional):
  ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 ubuntu@<ip>:/path/to/best.pt ./weights/
  ```
10. **Ollama placement** — Simplest: install Ollama **on the same machine as FastAPI** (all on the VM, or all local after copying weights). Splitting API vs LLM across machines adds networking complexity.
11. **Safe API access** — SSH port-forward from your laptop:
  ```bash
    ssh -i ~/.ssh/id_ed25519 -p <port> -L 8000:127.0.0.1:8000 ubuntu@<instance-ip>
  ```
    On the VM: `uvicorn main:app --host 127.0.0.1 --port 8000` — then open `http://127.0.0.1:8000` locally.

## Modules

- `main.py` — FastAPI routes and in-memory state
- `yolo_model.py` — Ultralytics load + predict
- `vision.py` — detection → phrases
- `sensors.py` — thresholds → summary
- `llm.py` — Ollama `/api/generate` + JSON parsing
- `train.py` — training CLI

## Notes

- A stock `yolov8n.pt` predicts **COCO** classes, not the four leaf labels. Use your fine-tuned `best.pt` for meaningful plant-health boxes, or `DEMO_FAKE_VISION=1` for HTTP demos without weights.
- `/analysis` returns **502** if Ollama is down or the model output is not valid JSON.

