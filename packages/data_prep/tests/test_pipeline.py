"""End-to-end pipeline tests: synthetic data/raw in tmp_path, real pHash, real manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import qsd_data_prep
from qsd_common.manifest import load_manifest
from qsd_data_prep.config import DataPrepConfig

CATEGORIES = [
    {"id": 0, "name": "boat"},
    {"id": 1, "name": "dolphin"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "shark"},
]

RNG = np.random.default_rng(7)


def _clip_frames(n: int) -> list[np.ndarray]:
    """One 'video clip': a base image plus near-identical noisy variants (near-dups)."""
    base = RNG.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    frames = [base]
    for _ in range(n - 1):
        noise = RNG.integers(-4, 5, size=base.shape, dtype=np.int16)
        frames.append(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return frames


@pytest.fixture()
def synthetic_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic data/raw pool: 30 clips x 3 frames, mixed classes + background clips.

    Monkeypatches data_dir/manifests_dir everywhere the pipeline resolves them.
    """
    data = tmp_path / "data"
    manifests = tmp_path / "manifests"
    raw = data / "raw"
    images_dir = raw / "images"
    images_dir.mkdir(parents=True)
    manifests.mkdir()

    import qsd_common.io as io
    import qsd_common.manifest as manifest_mod

    for mod in (qsd_data_prep, manifest_mod):
        monkeypatch.setattr(mod, "data_dir", lambda: data)
    monkeypatch.setattr(qsd_data_prep, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(io, "data_dir", lambda: data)
    monkeypatch.setattr(io, "manifests_dir", lambda: manifests)

    images, annotations = [], []
    ann_id = 1
    image_id = 0
    for clip in range(30):
        for frame in _clip_frames(3):
            name = f"sharkspotting_{image_id + 1:06d}.jpg"
            Image.fromarray(frame).save(images_dir / name, quality=95)
            images.append({"id": image_id, "file_name": name, "height": 96, "width": 96})
            if clip % 6 == 0:
                cids = []  # background-only clip
            elif clip % 5 == 0:
                cids = [0, 2]  # rare boat clips
            elif clip % 2 == 0:
                cids = [3, 2, 2]
            else:
                cids = [1, 3, 2]
            for cid in cids:
                annotations.append(
                    {"id": ann_id, "image_id": image_id, "category_id": cid,
                     "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0, "segmentation": []}
                )
                ann_id += 1
            image_id += 1

    (raw / "annotations.coco.json").write_text(
        json.dumps(
            {"info": {"description": "synthetic"}, "licenses": [],
             "categories": CATEGORIES, "images": images, "annotations": annotations}
        )
    )
    return tmp_path


def _config(**overrides) -> DataPrepConfig:
    defaults = dict(name="test", seed=42, phash_threshold=8, calib_size=16)
    return DataPrepConfig(**{**defaults, **overrides})


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_pipeline_idempotent_byte_identical(synthetic_raw: Path) -> None:
    manifest_path = qsd_data_prep.run(_config())
    first = manifest_path.read_bytes()
    processed = synthetic_raw / "data" / "processed"
    first_tree = _tree_bytes(processed)

    qsd_data_prep.run(_config(), force=True)
    assert manifest_path.read_bytes() == first
    assert _tree_bytes(processed) == first_tree


def test_pipeline_refuses_overwrite_without_force(synthetic_raw: Path) -> None:
    qsd_data_prep.run(_config())
    with pytest.raises(SystemExit, match="--force"):
        qsd_data_prep.run(_config())


def test_manifest_contents(synthetic_raw: Path) -> None:
    manifest_path = qsd_data_prep.run(_config(), materialize=False)
    manifest = load_manifest(manifest_path)  # load_manifest re-runs verify_manifest

    assert len(manifest.images) == 90
    assert manifest.meta["phash"]["threshold"] == 8
    assert manifest.meta["seed"] == 42
    assert manifest.meta["splits"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert "imagehash_version" in manifest.meta["phash"]

    # Near-dup frames of the same clip never straddle splits (also covered by verify,
    # but assert on real pHash grouping here: frames 3k..3k+2 share a clip).
    for clip_start in range(0, 90, 3):
        splits = {manifest.images[i].split for i in range(clip_start, clip_start + 3)}
        assert len(splits) == 1

    calib = manifest.calib_images()
    assert len(calib) == 16
    assert all(e.split == "train" for e in calib)
    assert len({e.group_id for e in calib}) == len(calib)

    summary = manifest.summary
    assert summary["images_per_split"]["train"] > summary["images_per_split"]["val"]
    for split, per_class in summary["annotations_per_split_per_class"].items():
        for name, count in per_class.items():
            assert count > 0, f"{name} missing from {split}"


def test_materialized_tree_matches_manifest(synthetic_raw: Path) -> None:
    from qsd_common.manifest import verify_materialized

    qsd_data_prep.run(_config())
    manifest = load_manifest(synthetic_raw / "manifests" / "split_manifest.json")
    processed = synthetic_raw / "data" / "processed"
    for split in ("train", "val", "test", "calib"):
        verify_materialized(split, manifest=manifest, processed_dir=processed)
