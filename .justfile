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

# Sync, pull data, and run the data-prep stage.
[group('stages')]
data-prep:
    uv sync --directory packages/data_prep
    just dop="{{dop}}" pull
    {{dop}} uv run --directory packages/data_prep data-prep

# --- Stage 2: training (cloud GPU) ---

# Sync, pull data, and train one framework: `just train ultralytics|roboflow|hf`.
[group('stages')]
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
[group('stages')]
edge-setup:
    uv venv --directory packages/edge --system-site-packages --python /usr/bin/python3

# Sync (keeping system packages), pull data, and run the edge stage.
[group('stages')]
edge:
    uv sync --inexact --directory packages/edge
    just dop="{{dop}}" pull
    {{dop}} uv run --directory packages/edge edge

# --- DVC (Cloudflare R2) ---

# Pull data/model artifacts from the R2 remote (R2 creds injected by Doppler).
[group('dvc')]
pull:
    {{dop}} uvx dvc pull

# Push data/model artifacts to the R2 remote (R2 creds injected by Doppler).
[group('dvc')]
push:
    {{dop}} uvx dvc push

# Show DVC artifact status.
[group('dvc')]
dvc-status:
    uvx dvc status

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
