"""Hermetic tests for the verify-splits entrypoint (qsd_common.verify)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Reuse the synthetic-manifest builder and the raw_dir/entries fixtures from test_manifest
# (pytest's prepend import mode puts the tests dir on sys.path, so it imports by basename).
from test_manifest import _manifest, entries, raw_dir  # noqa: F401

from qsd_common import materialize_splits
from qsd_common.verify import main


def _run(monkeypatch: pytest.MonkeyPatch, manifest_path: Path, processed: Path, split: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify-splits", "--split", split,
         "--manifest", str(manifest_path), "--processed-dir", str(processed)],
    )
    main()


def test_main_verifies_all_splits(
    tmp_path: Path, raw_dir: Path, entries: list, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    manifest = _manifest(entries)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    processed = tmp_path / "processed"
    materialize_splits(manifest, raw_dir=raw_dir, out_dir=processed)

    _run(monkeypatch, manifest_path, processed, "all")  # passes → no exception
    _run(monkeypatch, manifest_path, processed, "val")  # single split also works


def test_main_fails_loud_on_missing_image(
    tmp_path: Path, raw_dir: Path, entries: list, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    manifest = _manifest(entries)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    processed = tmp_path / "processed"
    materialize_splits(manifest, raw_dir=raw_dir, out_dir=processed)

    (processed / "test" / "images" / "img_004.jpg").unlink()
    with pytest.raises(AssertionError, match="missing"):
        _run(monkeypatch, manifest_path, processed, "all")
