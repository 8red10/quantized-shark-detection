# Quantized Shark Detection
Explores the accuracy-latency-power Pareto frontier derived from quantizing object detection models of varying architectures and deploying them on the edge. 

All experiment stages can be found in this repository.

## 1. Data Preparation
Creates the train/val/test splits while ensuring near-duplicate images are kept to a single split to help prevent memorization. Also, identifies the calibration set for INT8 quantization.

Hardware = local CPU

## 2. Training
Trains each model using frameworks and pipelines tuned to that model architecture. After training, exports each model to ONNX for compatibility with quantization. 

Hardware = cloud GPU

## 3. Edge Deployment
Quantizes and benchmarks each model to record accuracy, latency and power when deployed on the edge.

Hardware = Jetson Orin Nano

# Repository Layout

This is a **monorepo of independent `uv` projects** — *not* a `uv` workspace. The three
stages run on different machines (local CPU, cloud GPU, Jetson) with incompatible
dependency stacks (JetPack/TensorRT on the edge; three conflicting training frameworks),
so a single shared lockfile is impossible. Instead **each stage has its own
`pyproject.toml` + `uv.lock` + `.venv`**, and they share code through an **editable path
dependency on `packages/common`**. The repo root has *no* `[project]` table and *no*
`[tool.uv.workspace]`, so it is neither a package nor a workspace.

```
qsd/
├── pyproject.toml            # shared ruff/pytest config ONLY (not a package/workspace)
├── .gitignore
├── .dvc/  .dvcignore         # data/models pulled from Cloudflare R2 via DVC
├── README.md
│
├── data/                     # dvc-tracked (gitignored)
├── models/                   # dvc-tracked — trained weights + ONNX exports
├── manifests/                # small split/calibration manifests (committed)
├── configs/                  # shared experiment configs (yaml)
│
└── packages/
    ├── common/               # shared library — qsd-common (imported, never run)
    │   └── src/qsd_common/   #   io.py, config.py, utils.py, onnx.py
    │
    ├── data_prep/            # Stage 1 · qsd-data-prep   · own uv.lock · local CPU
    │   └── src/qsd_data_prep/
    │
    ├── training/             # Stage 2 · one independent project per framework
    │   ├── ultralytics/      #   qsd-train-ultralytics · own uv.lock · cloud GPU
    │   ├── roboflow/         #   qsd-train-roboflow     · own uv.lock · cloud GPU
    │   └── hf/               #   qsd-train-hf           · own uv.lock · cloud GPU
    │       └── src/qsd_train_hf/
    │
    └── edge/                 # Stage 3 · qsd-edge · own uv.lock (glue only) · Jetson
        └── src/qsd_edge/     #   TensorRT/torch come from JetPack, not the lockfile
```

> Import packages are `qsd_`-prefixed (`qsd_train_ultralytics`, …) so they never shadow
> the real `ultralytics` / `roboflow` PyPI packages.

# Setup

Each machine clones the whole repo but only sets up its own stage:

```bash
# Stage 1 — local CPU
cd packages/data_prep       && uv sync && dvc pull && uv run data-prep

# Stage 2 — cloud GPU (pick the framework)
cd packages/training/ultralytics && uv sync && dvc pull && uv run train-ultralytics
cd packages/training/roboflow    && uv sync && dvc pull && uv run train-roboflow
cd packages/training/hf          && uv sync && dvc pull && uv run train-hf

# Stage 3 — Jetson Orin Nano (venv sees JetPack's TensorRT/torch)
cd packages/edge
uv venv --system-site-packages --python /usr/bin/python3
uv sync --inexact           # --inexact: keep the system-provided packages
dvc pull && uv run edge
```

Configure the DVC remote once (per your Cloudflare R2 bucket):

```bash
dvc remote add -d r2 s3://<bucket>/<path>
dvc remote modify r2 endpointurl https://<account>.r2.cloudflarestorage.com
# credentials via env: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (R2 tokens)
```