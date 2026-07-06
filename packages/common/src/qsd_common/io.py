"""Filesystem helpers for locating repo-level, DVC-managed artifact directories.

The repo is cloned whole on every machine, so stage code can resolve shared
paths (``data/``, ``models/``, ``manifests/``) relative to the repo root
regardless of which package it runs from.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the monorepo root by walking up until a ``packages/`` dir is found.

    Falls back to a ``.git`` marker. Raises if neither is found.
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "packages").is_dir() or (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not locate repo root from {here}")


def data_dir() -> Path:
    return repo_root() / "data"


def models_dir() -> Path:
    return repo_root() / "models"


def manifests_dir() -> Path:
    return repo_root() / "manifests"


def clear_dir(path: Path, *, attempts: int = 5) -> None:
    """Recursively remove ``path``, retrying transient races.

    On macOS, Finder/Spotlight can drop a fresh ``.DS_Store`` into a directory while
    ``shutil.rmtree`` is deleting it bottom-up, so the final ``rmdir`` fails with ENOTEMPTY even
    though we just emptied it. A few bounded retries clear that; ``rmtree`` on the already-shrunk
    tree is safe to repeat. This keeps ``--force`` a reliable rebuild/recovery path.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)
