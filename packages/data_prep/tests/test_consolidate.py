"""Tests for qsd_data_prep.consolidate — hermetic, on a synthetic roboflow-split fixture.

Never touches the real 2.4 GB dataset: a tiny fake export is built in ``tmp_path`` and
``data_dir`` is monkeypatched, so the whole suite runs in-memory-fast. The headline test is
idempotency: two consolidations of the same source must yield byte-identical output, which is
what keeps the DVC content hash stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsd_data_prep import consolidate as C

# Real Roboflow category layout in miniature: a dummy ``sharks`` umbrella at id 0 (dropped),
# then the real classes whose original id order yields boat, dolphin, person, shark.
CATEGORIES = [
    {"id": 0, "name": "sharks", "supercategory": "none"},
    {"id": 1, "name": "boat", "supercategory": "sharks"},
    {"id": 2, "name": "dolphin", "supercategory": "sharks"},
    {"id": 3, "name": "person", "supercategory": "sharks"},
    {"id": 4, "name": "shark", "supercategory": "sharks"},
]

# Per-split source export dates (distinct, out of order) to prove the derived date is max().
SPLIT_DATES = {
    "train": "2026-07-05T05:25:32+00:00",
    "valid": "2026-07-03T00:00:00+00:00",
    "test": "2026-07-04T00:00:00+00:00",
}
EXPECTED_EXPORT_DATE = "2026-07-05T05:25:32+00:00"  # == max(SPLIT_DATES)


def _write_split(
    source_root: Path,
    split: str,
    *,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] = CATEGORIES,
    date_created: str | None = None,
) -> None:
    """Write one synthetic ``<split>/`` with its images and ``_annotations.coco.json``."""
    split_dir = source_root / split
    split_dir.mkdir(parents=True)
    for img in images:
        # Content need not be a valid JPEG — consolidate only copies files and reads dims from JSON.
        (split_dir / img["file_name"]).write_bytes(f"pixels::{split}::{img['file_name']}".encode())
    coco = {
        "info": {"version": "3", "date_created": date_created or SPLIT_DATES[split]},
        "licenses": [{"id": 1, "name": "cc"}],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


def _img(img_id: int, name: str, *, extra_name: str | None = None) -> dict:
    d = {"id": img_id, "file_name": name, "height": 10, "width": 20}
    if extra_name is not None:
        d["extra"] = {"name": extra_name}
    return d


def _ann(ann_id: int, image_id: int, category_id: int) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [1, 2, 3, 4],
        "area": 12,
        "iscrowd": 0,
        "segmentation": [],
    }


@pytest.fixture
def fake_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a 3-split synthetic export under tmp_path and point data_dir at it.

    Splits use per-split restarting image/annotation ids (as Roboflow does), so the test also
    exercises the global re-indexing. Totals: 4 images, 6 annotations.
    """
    source_root = tmp_path / C.SOURCE_DIR
    _write_split(
        source_root,
        "train",
        images=[_img(0, "t0.jpg", extra_name="orig_t0"), _img(1, "t1.jpg")],
        annotations=[_ann(0, 0, 1), _ann(1, 0, 4), _ann(2, 1, 3)],  # boat, shark, person
    )
    _write_split(
        source_root,
        "valid",
        images=[_img(0, "v0.jpg")],
        annotations=[_ann(0, 0, 2)],  # dolphin
    )
    _write_split(
        source_root,
        "test",
        images=[_img(0, "x0.jpg")],
        annotations=[_ann(0, 0, 1), _ann(1, 0, 1)],  # boat, boat
    )
    monkeypatch.setattr(C, "data_dir", lambda: tmp_path)
    return tmp_path


def _run(fake_data_dir: Path, **kwargs) -> dict:
    C.consolidate(force=True, expected_images=4, expected_annotations=6, **kwargs)
    return json.loads((fake_data_dir / "raw" / "annotations.coco.json").read_text())


def test_idempotent_byte_identical_output(fake_data_dir: Path) -> None:
    out = fake_data_dir / "raw" / "annotations.coco.json"

    _run(fake_data_dir)
    first = out.read_bytes()
    _run(fake_data_dir)  # second consolidation of the same source
    second = out.read_bytes()

    assert first == second, "consolidated annotations.coco.json is not byte-identical across runs"


def test_export_date_is_source_derived_not_wallclock(fake_data_dir: Path) -> None:
    doc = _run(fake_data_dir)
    assert doc["info"]["source_export_date"] == EXPECTED_EXPORT_DATE  # max() of the split dates
    assert doc["info"]["source_version"] == "3"
    assert "date_created" not in doc["info"]  # intentionally omitted (would misname the value)


def test_filenames_are_sequential_zero_padded(fake_data_dir: Path) -> None:
    doc = _run(fake_data_dir)
    names = [im["file_name"] for im in doc["images"]]
    assert names == [
        "sharkspotting_000001.jpg",
        "sharkspotting_000002.jpg",
        "sharkspotting_000003.jpg",
        "sharkspotting_000004.jpg",
    ]
    for im in doc["images"]:
        assert (fake_data_dir / "raw" / "images" / im["file_name"]).exists()


def test_ids_reindexed_globally_and_contiguous(fake_data_dir: Path) -> None:
    doc = _run(fake_data_dir)
    assert [im["id"] for im in doc["images"]] == [0, 1, 2, 3]
    assert [a["id"] for a in doc["annotations"]] == [1, 2, 3, 4, 5, 6]  # 1-based, no per-split restart
    valid_ids = {im["id"] for im in doc["images"]}
    assert all(a["image_id"] in valid_ids for a in doc["annotations"])


def test_category_remap_drops_dummy_and_is_contiguous(fake_data_dir: Path) -> None:
    doc = _run(fake_data_dir)
    assert [(c["id"], c["name"]) for c in doc["categories"]] == [
        (0, "boat"),
        (1, "dolphin"),
        (2, "person"),
        (3, "shark"),
    ]
    assert all(a["category_id"] in {0, 1, 2, 3} for a in doc["annotations"])
    # train's shark(orig 4) -> 3, person(orig 3) -> 2; valid's dolphin(orig 2) -> 1.
    per_image = {a["image_id"]: [] for a in doc["annotations"]}
    for a in doc["annotations"]:
        per_image[a["image_id"]].append(a["category_id"])
    assert sorted(per_image[0]) == [0, 3]  # boat, shark on first train image
    assert per_image[2] == [1]  # dolphin on the valid image (global id 2)


def test_provenance_preserved(fake_data_dir: Path) -> None:
    doc = _run(fake_data_dir)
    first = doc["images"][0]
    assert first["extra"]["roboflow_file"] == "t0.jpg"
    assert first["extra"]["source_split"] == "train"
    assert first["extra"]["name"] == "orig_t0"
    assert first["license"] == C.LICENSE["id"]
    # image lacking source extra.name -> None, not a crash
    assert doc["images"][1]["extra"]["name"] is None


def test_existing_nonempty_raw_without_force_exits(fake_data_dir: Path) -> None:
    raw = fake_data_dir / "raw"
    raw.mkdir()
    (raw / "stale.txt").write_text("leftover")
    with pytest.raises(SystemExit):
        C.consolidate(expected_images=4, expected_annotations=6)


def test_force_clears_and_rebuilds_over_existing(fake_data_dir: Path) -> None:
    raw = fake_data_dir / "raw"
    raw.mkdir()
    (raw / "stale.txt").write_text("leftover")
    _run(fake_data_dir)  # force=True
    assert not (raw / "stale.txt").exists()  # prior contents cleared
    assert (raw / "annotations.coco.json").exists()


def test_category_drift_between_splits_exits(fake_data_dir: Path) -> None:
    # Rewrite the valid split with a different taxonomy; consolidate must fail loud.
    source_root = fake_data_dir / C.SOURCE_DIR
    import shutil

    shutil.rmtree(source_root / "valid")
    _write_split(
        source_root,
        "valid",
        images=[_img(0, "v0.jpg")],
        annotations=[_ann(0, 0, 1)],
        categories=[
            {"id": 0, "name": "sharks", "supercategory": "none"},
            {"id": 1, "name": "whale", "supercategory": "sharks"},
        ],
    )
    with pytest.raises(SystemExit):
        C.consolidate(force=True, expected_images=4, expected_annotations=6)
