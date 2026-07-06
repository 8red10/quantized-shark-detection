"""Hermetic tests for qsd_common.manifest (no real dataset, everything in tmp_path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsd_common.manifest import (
    ImageEntry,
    SplitManifest,
    load_manifest,
    materialize_splits,
    verify_manifest,
    verify_materialized,
)

CATEGORIES = [{"id": 0, "name": "boat"}, {"id": 1, "name": "shark"}]


def _entry(image_id: int, group_id: int, split: str, *, is_calib: bool = False) -> ImageEntry:
    return ImageEntry(
        image_id=image_id,
        file_name=f"img_{image_id:03d}.jpg",
        phash=f"{image_id:016x}",
        group_id=group_id,
        primary_class="shark" if image_id % 2 else None,
        split=split,
        is_calib=is_calib,
    )


def _manifest(entries: list[ImageEntry]) -> SplitManifest:
    return SplitManifest(
        meta={"manifest_version": 1},
        summary={
            "images_per_split": {
                s: sum(1 for e in entries if e.split == s) for s in ("train", "val", "test")
            }
        },
        images=entries,
    )


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """A tiny fake data/raw: 6 one-byte 'images' + a COCO json with 3 annotations."""
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    images, annotations = [], []
    for i in range(6):
        name = f"img_{i:03d}.jpg"
        (raw / "images" / name).write_bytes(bytes([i]))
        images.append({"id": i, "file_name": name, "height": 4, "width": 4})
    for ann_id, (img, cat) in enumerate([(0, 1), (2, 0), (4, 1)], start=1):
        annotations.append(
            {"id": ann_id, "image_id": img, "category_id": cat, "bbox": [0, 0, 1, 1],
             "area": 1, "iscrowd": 0, "segmentation": []}
        )
    (raw / "annotations.coco.json").write_text(
        json.dumps(
            {"info": {"description": "test"}, "licenses": [], "categories": CATEGORIES,
             "images": images, "annotations": annotations}
        )
    )
    return raw


@pytest.fixture()
def entries() -> list[ImageEntry]:
    return [
        _entry(0, 0, "train", is_calib=True),
        _entry(1, 0, "train"),
        _entry(2, 1, "train"),
        _entry(3, 2, "val"),
        _entry(4, 3, "test"),
        _entry(5, 4, "test"),
    ]


def test_load_manifest_round_trip(tmp_path: Path, entries: list[ImageEntry]) -> None:
    manifest = _manifest(entries)
    path = tmp_path / "split_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    loaded = load_manifest(path)
    assert loaded == manifest
    assert [e.image_id for e in loaded.by_split("train")] == [0, 1, 2]
    assert [e.image_id for e in loaded.calib_images()] == [0]


def test_verify_manifest_rejects_group_straddling_splits(entries: list[ImageEntry]) -> None:
    entries[1] = _entry(1, 0, "val")  # group 0 now spans train and val
    with pytest.raises(AssertionError, match="straddles"):
        verify_manifest(_manifest(entries))


def test_verify_manifest_rejects_calib_outside_train(entries: list[ImageEntry]) -> None:
    entries[3] = _entry(3, 2, "val", is_calib=True)
    with pytest.raises(AssertionError, match="calib"):
        verify_manifest(_manifest(entries))


def test_verify_manifest_rejects_duplicate_ids(entries: list[ImageEntry]) -> None:
    entries.append(_entry(0, 0, "train"))
    with pytest.raises(AssertionError, match="duplicate"):
        verify_manifest(_manifest(entries))


def test_verify_manifest_rejects_summary_mismatch(entries: list[ImageEntry]) -> None:
    manifest = _manifest(entries)
    manifest.summary["images_per_split"]["train"] += 1
    with pytest.raises(AssertionError, match="images_per_split"):
        verify_manifest(manifest)


def test_materialize_splits_builds_correct_tree(
    tmp_path: Path, raw_dir: Path, entries: list[ImageEntry]
) -> None:
    out = tmp_path / "processed"
    materialize_splits(_manifest(entries), raw_dir=raw_dir, out_dir=out)

    assert sorted(p.name for p in (out / "train" / "images").iterdir()) == [
        "img_000.jpg", "img_001.jpg", "img_002.jpg"
    ]
    assert [p.name for p in (out / "calib" / "images").iterdir()] == ["img_000.jpg"]
    # image bytes are true copies of raw
    assert (out / "test" / "images" / "img_004.jpg").read_bytes() == bytes([4])

    train_coco = json.loads((out / "train" / "annotations.coco.json").read_text())
    assert [im["id"] for im in train_coco["images"]] == [0, 1, 2]
    assert [a["id"] for a in train_coco["annotations"]] == [1, 2]  # anns of images 0 and 2
    assert train_coco["categories"] == CATEGORIES
    assert train_coco["info"]["split"] == "train"
    assert train_coco["info"]["num_images"] == 3
    assert "train split" in train_coco["info"]["description"]

    val_coco = json.loads((out / "val" / "annotations.coco.json").read_text())
    assert val_coco["annotations"] == []  # image 3 has no annotations
    assert "val split" in val_coco["info"]["description"]


def test_materialize_splits_deterministic_and_force(
    tmp_path: Path, raw_dir: Path, entries: list[ImageEntry]
) -> None:
    manifest = _manifest(entries)
    out = tmp_path / "processed"
    materialize_splits(manifest, raw_dir=raw_dir, out_dir=out)
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}

    with pytest.raises(SystemExit, match="force"):
        materialize_splits(manifest, raw_dir=raw_dir, out_dir=out)

    materialize_splits(manifest, raw_dir=raw_dir, out_dir=out, force=True)
    second = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    assert first == second  # byte-identical rebuild → stable DVC hashes


def test_verify_materialized_passes_and_fails_loud(
    tmp_path: Path, raw_dir: Path, entries: list[ImageEntry]
) -> None:
    manifest = _manifest(entries)
    out = tmp_path / "processed"
    materialize_splits(manifest, raw_dir=raw_dir, out_dir=out)

    for split in ("train", "val", "test", "calib"):
        verify_materialized(split, manifest=manifest, processed_dir=out)

    (out / "test" / "images" / "img_004.jpg").unlink()
    with pytest.raises(AssertionError, match="missing"):
        verify_materialized("test", manifest=manifest, processed_dir=out)

    (out / "val" / "images" / "stray.jpg").write_bytes(b"x")
    with pytest.raises(AssertionError, match="unexpected"):
        verify_materialized("val", manifest=manifest, processed_dir=out)
