# Quantized Shark Detection
Explores the accuracy-latency-power Pareto frontier derived from quantizing object detection models of varying architectures and deploying them on the edge. 

All experiment stages can be found in this repository.

## 1. Data Preparation
Creates the train/val/test splits while ensuring near-duplicate images are kept to a single split to help prevent memorization. Also, identifies the calibration set for INT8 quantization.

Hardware = local CPU

<details><summary>How <code>data/raw</code> was built from the Roboflow export</summary>

The source is a Roboflow COCO export (*SharkSpotting v3*) whose `train/valid/test` splits
were made **without** near-duplicate grouping, so those splits are discarded — we re-split
later (above) with near-dup awareness. `just consolidate-raw` turns the export
(`data/roboflow-split/`) into a single split-free pool at `data/raw/`:

- **Merges** all three splits into `data/raw/images/` + `data/raw/annotations.coco.json` —
  **4656 images / 8857 annotations**.
- **Renames** images to `sharkspotting_000001.jpg … sharkspotting_004656.jpg`, stripping
  Roboflow's `.rf.<hash>` suffixes.
- **Cleans the taxonomy** to a contiguous 0-indexed 4-class set — `0 boat, 1 dolphin,
  2 person, 3 shark` — dropping Roboflow's unused dummy `id 0` supercategory.
- **Re-indexes** image and annotation IDs globally (each Roboflow split restarts them) and
  keeps provenance per image in `extra` (`name`, `roboflow_file`, `source_split`). Roboflow
  cruft (per-split `info`/`licenses`, READMEs) is stripped; CC BY 4.0 + the source URL are
  retained in `info`.

This is a **one-time** step, but it is also idempotent for reproducibility. After 
verifying, publish the pool with `dvc add data/raw && just push`; from then on 
`just pull-raw` fetches it and re-running the consolidation is unnecessary. 

</details>

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

Each machine clones the whole repo but only sets up its own stage. Common tasks are
wrapped in a root [`justfile`](.justfile) — run `just` from the repo root to list them
(`just`, like `dvc`, must run from the repo root). Each recipe handles
`uv sync → dvc pull → uv run` for its stage.

```bash
uv tool install rust-just     # provides `just` on the cloud GPU / Jetson (aarch64)
just --list                   # discover recipes

just data-prep                # Stage 1 — local CPU
just train ultralytics        # Stage 2 — cloud GPU (or: roboflow | hf)
just edge-setup && just edge  # Stage 3 — Jetson: bootstrap venv once, then run
```

The Jetson's `edge-setup` recipe creates the venv with `--system-site-packages` so it can
see JetPack's TensorRT/torch; the `edge` recipe then syncs with `--inexact` (keeping those
system packages) and runs. Cross-stage dev tasks are also available: `just lint`,
`just fmt`, `just lock` (re-lock every stage), `just test`, `just check`.

<details><summary>Raw commands (no <code>just</code>)</summary>

```bash
# Stage 1 — local CPU
cd packages/data_prep && uv sync && dvc pull && uv run data-prep
# Stage 2 — cloud GPU (pick the framework)
cd packages/training/<ultralytics|roboflow|hf> && uv sync && dvc pull && uv run train-<framework>
# Stage 3 — Jetson Orin Nano (venv sees JetPack's TensorRT/torch)
cd packages/edge && uv venv --system-site-packages --python /usr/bin/python3
uv sync --inexact && dvc pull && uv run edge
```
</details>

The DVC remote `r2` (`s3://qsd/v1`, `region = auto`) is committed to `.dvc/config`. The
account-specific **endpoint** and the **credentials** are kept out of git — both come from
Doppler. Once per machine, populate the endpoint into the gitignored `.dvc/config.local`:

```bash
just dvc-setup   # writes R2_ENDPOINT_URL (from Doppler) into .dvc/config.local
```

After that, `just pull` / `just push` work: DVC merges `config` + `config.local` for the
endpoint and reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from the Doppler-injected
environment (`region = auto` is committed; nothing secret is stored in git).

# Secrets

Secrets (R2 credentials, Roboflow / HuggingFace tokens) are managed centrally with
[Doppler](https://docs.doppler.com/) — nothing sensitive lives in git. The `just` recipes
inject them by prefixing secret-touching commands with `doppler run --` (the overridable
`dop` variable), so `dvc pull`/`push` and the stage runs get their env vars automatically.
Bypass Doppler on a machine that isn't using it with `just dop='' <recipe>`.

Secrets are scoped per stage (least privilege) in one Doppler project `qsd`: a root config
`prd` holds the shared R2 credentials plus the Telegram bot secrets (`TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`) so any stage can send notifications; a `prd_training` branch config adds
`ROBOFLOW_API_KEY` and `HF_TOKEN`. Each machine authenticates once:

| Machine | Install Doppler | Auth |
|---|---|---|
| Mac (data prep) | `brew install dopplerhq/cli/doppler` | `doppler login && doppler setup -p qsd -c prd_data` |
| Cloud GPU (training) | `curl -Ls https://cli.doppler.com/install.sh \| sh` | `export DOPPLER_TOKEN=<prd_training service token>` |
| Jetson (edge) | Doppler install script (arm64) | `export DOPPLER_TOKEN=<prd_edge service token>` |

The headless machines use read-only **service tokens** (they can't run interactive
`doppler login`); store the token via the machine's own env mechanism, never in the repo.
Run `just secrets` to see which keys Doppler will inject (values masked).

# Notifications

`qsd-common` provides Telegram helpers (`qsd_common.notify`) so any stage can push alerts —
primarily to signal when an unattended cloud-GPU training run finishes. Credentials come
from Doppler (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`); sends are **best-effort** (failures
are logged, never raised, and time out), so a completed run is never killed by a notify
error. The API exposes `send_message` (auto-chunked), `send_photo` (plots), `send_document`
(reports/files), and a `notify_on_completion` context manager that alerts on success/failure:

```python
from qsd_common import notify_on_completion

with notify_on_completion("train-ultralytics") as tg:  # ✅/❌ alert on exit
    metrics = train(...)
    tg.send_photo("runs/pr_curve.png", caption="PR curve")
    tg.send_document("runs/results.csv", caption=f"mAP={metrics['map']:.3f}")
```

Because secrets come from Doppler, run training under it: `just train ultralytics` (which
wraps the run in `doppler run --`) delivers the alert automatically.