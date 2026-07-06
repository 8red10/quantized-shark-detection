# Quantized Shark Detection — task runner.
# Run from the repo root (like dvc). `just` or `just --list` shows all recipes.
# On other machines: `uv tool install rust-just` provides `just`.

train_frameworks := "ultralytics roboflow hf"

# Prefix that injects secrets from Doppler as env vars (R2/DVC, Roboflow, HF, ...).
# Override to bypass Doppler on a machine not using it: `just dop='' <recipe>`.
dop := "doppler run --"

# List all recipes.
default:
    @just --list

# --- Stage 1: data prep (local CPU) ---

# Sync, pull data, and run the data-prep stage (builds the manifest + data/processed).
[group('stage 1: data prep')]
data-prep *args:
    uv sync --directory packages/data_prep
    just dop="{{dop}}" pull-raw
    uv run --directory packages/data_prep data-prep {{args}}

# One-time: report grouping stats per pHash threshold (+ pair montages to eyeball).
[group('stage 1: data prep')]
explore-thresholds *args:
    uv sync --directory packages/data_prep
    just dop="{{dop}}" pull-raw
    uv run --directory packages/data_prep explore-thresholds {{args}}

# One-time: rebuild data/raw from the Roboflow export (data/roboflow-export).
[group('stage 1: data prep')]
consolidate-raw *args:
    uv sync --directory packages/data_prep
    just dop="{{dop}}" pull-rf
    uv run --directory packages/data_prep consolidate-raw {{args}}

# Open a dataset in the FiftyOne app: just visualize train|val|test|calib|raw|roboflow-{train,valid,test}.
[group('stage 1: data prep')]
visualize dataset="train" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{dataset}}" in
        raw)                  dvc_path=data/raw ;;
        train|val|test|calib) dvc_path=data/processed/{{dataset}} ;;
        roboflow-*)           dvc_path=data/roboflow-export ;;
        *) echo "unknown dataset '{{dataset}}'"; exit 1 ;;
    esac
    uv sync --directory packages/data_prep --group fiftyone
    {{dop}} dvc pull "$dvc_path"
    uv run --directory packages/data_prep visualize --dataset {{dataset}} {{args}}

# --- Stage 2: training (cloud GPU) ---

# Sync, pull data, and train one framework: `just train ultralytics|roboflow|hf`.
[group('stage 2: training')]
train framework:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{framework}}" in
        ultralytics|roboflow|hf) ;;
        *) echo "unknown framework '{{framework}}' (use: {{train_frameworks}})"; exit 1 ;;
    esac
    uv sync --directory packages/training/{{framework}}
    just dop="{{dop}}" pull
    {{dop}} uv run --directory packages/training/{{framework}} train-{{framework}}

# --- Stage 3: edge (Jetson Orin Nano) ---

# One-time (Jetson): create a venv that can see JetPack's TensorRT/torch.
[group('stage 3: edge')]
edge-setup:
    uv venv --directory packages/edge --system-site-packages --python /usr/bin/python3

# Sync (keeping system packages), pull data, and run the edge stage.
[group('stage 3: edge')]
edge:
    uv sync --inexact --directory packages/edge
    just dop="{{dop}}" pull
    {{dop}} uv run --directory packages/edge edge

# --- DVC (Cloudflare R2) ---

# One-time per machine: write the R2 endpoint (from Doppler) into gitignored .dvc/config.local.
[group('dvc')]
dvc-setup:
    {{dop}} sh -c 'dvc remote modify --local r2 endpointurl "$R2_ENDPOINT_URL"'

# Pull all data/model artifacts from the R2 remote (R2 creds injected by Doppler).
[group('dvc')]
pull:
    {{dop}} dvc pull

# Pull roboflow download from the R2 remote.
[group('dvc')]
pull-rf:
    {{dop}} dvc pull data/roboflow-export

# Pull raw dataset from the R2 remote.
[group('dvc')]
pull-raw:
    {{dop}} dvc pull data/raw

# Pull one materialized split only: `just pull-split train|val|test|calib`.
[group('dvc')]
pull-split split:
    {{dop}} dvc pull data/processed/{{split}}

# Track the materialized splits with DVC (run after `just data-prep`); commit the .dvc files.
[group('dvc')]
dvc-add-processed:
    dvc add data/processed/train data/processed/val data/processed/test data/processed/calib

# Push data/model artifacts to the R2 remote (R2 creds injected by Doppler).
[group('dvc')]
push:
    {{dop}} dvc push

# Show DVC artifact status.
[group('dvc')]
ds:
    dvc status

# --- Secrets (Doppler) ---

# Show which secrets Doppler will inject into recipes (values masked).
[group('secrets')]
secrets:
    doppler secrets

# --- Cross-stage dev tasks ---

# Lint every package.
[group('dev')]
lint:
    uvx ruff check packages

# Format every package and apply safe lint fixes.
[group('dev')]
fmt:
    uvx ruff format packages
    uvx ruff check --fix packages

# Regenerate every stage lockfile independently (keeps the per-stage isolation).
[group('dev')]
lock:
    #!/usr/bin/env bash
    set -euo pipefail
    for p in data_prep training/ultralytics training/roboflow training/hf edge; do
        echo "== lock $p =="
        uv lock --directory "packages/$p"
    done

# Run each stage's tests in its own env.
[group('dev')]
test:
    #!/usr/bin/env bash
    set -euo pipefail
    for p in common data_prep training/ultralytics training/roboflow training/hf edge; do
        echo "== test $p =="
        uv run --directory "packages/$p" python -m pytest -q || true
    done

# Lint + test.
[group('dev')]
check: lint test

# Send a test message with telegram.
[group('dev')]
tg-send:
    {{dop}} uv run --directory packages/common python -c "from qsd_common import send_message; print(send_message('QSD test ✅'))"
