"""Consolidate the Roboflow COCO export into a single split-free ``data/raw`` pool.

The Roboflow export (``data/roboflow-split/{train,valid,test}/``) was split without
near-duplicate grouping in mind, so the splits are discarded here. This step merges all
three into one flat image directory plus one consolidated COCO annotations file, ready
for the near-dup-aware splitting that the data-prep stage performs later.

What it does:
  * copies every image into ``data/raw/images/`` under a consistent, hash-free name
    (``sharkspotting_000001.jpg`` … ``sharkspotting_004656.jpg``),
  * re-indexes image and annotation IDs globally (each Roboflow split restarts them),
  * cleans the category taxonomy to a contiguous 0-indexed set, dropping Roboflow's
    unused dummy ``id 0`` supercategory,
  * strips Roboflow cruft (per-split ``info``/``licenses``, READMEs) while preserving
    provenance (source URL, CC BY 4.0 license, original filename per image),
  * writes ``data/raw/annotations.coco.json`` and verifies the result (fail-loud).

Run via the ``consolidate-raw`` console script (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from qsd_common import data_dir, get_logger

log = get_logger(__name__)

SOURCE_DIR = "roboflow-split"
SPLITS = ("train", "valid", "test")
FILENAME_STEM = "sharkspotting"
PAD_WIDTH = 6

# Expected size of the consolidated pool; used by _verify as a fail-loud invariant on the real
# dataset. Exposed as constants (and _verify params) so tests can target synthetic fixtures.
EXPECTED_IMAGES = 4656
EXPECTED_ANNOTATIONS = 8857

# Deterministic fallback for the consolidated export date when the source export omits one.
# Never wall-clock time: the output must be byte-identical across runs (see module docstring).
FALLBACK_EXPORT_DATE = "1970-01-01T00:00:00+00:00"

# Provenance (from README.dataset.txt / README.roboflow.txt).
SOURCE_URL = "https://universe.roboflow.com/grad-1rom5/sharkspotting-akpys"
LICENSE = {"id": 1, "url": "https://creativecommons.org/licenses/by/4.0/", "name": "CC BY 4.0"}

# The one real class umbrella Roboflow injects at id 0 is unused by any annotation; drop it.
DUMMY_CATEGORY_NAMES = {"sharks"}


def _build_category_remap(categories: list[dict]) -> tuple[dict[int, int], list[dict]]:
    """Map old Roboflow category ids to a clean, contiguous 0-indexed taxonomy.

    Drops the unused dummy supercategory (id 0, ``sharks``) and re-numbers the remaining
    classes by their original id order, which yields ``boat, dolphin, person, shark``.
    Returns ``(old_id -> new_id, new_categories)``.
    """
    kept = [c for c in categories if c["name"] not in DUMMY_CATEGORY_NAMES]
    kept.sort(key=lambda c: c["id"])
    old_to_new: dict[int, int] = {}
    new_categories: list[dict] = []
    for new_id, cat in enumerate(kept):
        old_to_new[cat["id"]] = new_id
        new_categories.append({"id": new_id, "name": cat["name"], "supercategory": "none"})
    return old_to_new, new_categories


def _clear_dir(path: Path, *, attempts: int = 5) -> None:
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


def consolidate(
    *,
    force: bool = False,
    expected_images: int = EXPECTED_IMAGES,
    expected_annotations: int = EXPECTED_ANNOTATIONS,
) -> None:
    source_root = data_dir() / SOURCE_DIR
    raw_root = data_dir() / "raw"
    images_out = raw_root / "images"

    if raw_root.exists() and any(raw_root.iterdir()):
        if not force:
            raise SystemExit(
                f"{raw_root} already exists and is not empty. "
                f"Re-run with --force to rebuild it (output is deterministic, so a --force "
                f"rebuild is byte-identical and DVC-safe; it also recovers a partial run)."
            )
        log.info("--force given: clearing existing %s", raw_root)
        _clear_dir(raw_root)
    images_out.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    category_remap: dict[int, int] | None = None
    new_categories: list[dict] = []

    img_idx = 0  # global 0-based image id, also drives the sequential filename (idx+1)
    ann_id = 1  # global 1-based annotation id

    source_dates: list[str] = []  # per-split info.date_created, for a deterministic export date
    source_versions: list[str] = []  # per-split info.version, carried into output provenance

    for split in SPLITS:
        split_dir = source_root / split
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())

        source_info = coco.get("info", {})
        if source_info.get("date_created"):
            source_dates.append(source_info["date_created"])
        if source_info.get("version"):
            source_versions.append(str(source_info["version"]))

        if category_remap is None:
            category_remap, new_categories = _build_category_remap(coco["categories"])
        else:
            # Categories are identical across Roboflow splits; assert to catch surprises.
            remap, _ = _build_category_remap(coco["categories"])
            if remap != category_remap:
                raise SystemExit(f"Category set in split '{split}' differs from earlier splits.")

        old_to_new_image: dict[int, int] = {}
        for old_img in tqdm(
            sorted(coco["images"], key=lambda im: im["id"]),
            desc=f"copy {split}",
            unit="img",
        ):
            new_name = f"{FILENAME_STEM}_{img_idx + 1:0{PAD_WIDTH}d}.jpg"
            shutil.copy2(split_dir / old_img["file_name"], images_out / new_name)

            images.append(
                {
                    "id": img_idx,
                    "file_name": new_name,
                    "height": old_img["height"],
                    "width": old_img["width"],
                    "license": LICENSE["id"],
                    "extra": {
                        "name": old_img.get("extra", {}).get("name"),
                        "roboflow_file": old_img["file_name"],
                        "source_split": split,
                    },
                }
            )
            old_to_new_image[old_img["id"]] = img_idx
            img_idx += 1

        for ann in sorted(coco["annotations"], key=lambda a: a["id"]):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": old_to_new_image[ann["image_id"]],
                    "category_id": category_remap[ann["category_id"]],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": ann.get("segmentation", []),
                }
            )
            ann_id += 1

    # Finder can drop a fresh .DS_Store into raw_root/images_out while we work; it is also
    # excluded via .dvcignore, but prune it here so the on-disk tree matches the deterministic
    # file set exactly (self-healing guard — keeps a --force rebuild byte-identical and DVC-safe).
    for junk in raw_root.rglob(".DS_Store"):
        junk.unlink(missing_ok=True)

    _verify(
        images,
        annotations,
        new_categories,
        images_out,
        expected_images=expected_images,
        expected_annotations=expected_annotations,
    )

    # Derive a deterministic export date from the source (never wall-clock time) so the output is
    # byte-identical across runs and the DVC content hash stays stable. max() is order-independent
    # and picks the latest source export; source_version is likewise carried through for lineage.
    # Named source_export_date (not date_created) because it is the Roboflow export date, not a
    # creation timestamp.
    source_export_date = max(source_dates) if source_dates else FALLBACK_EXPORT_DATE
    source_version = max(source_versions) if source_versions else None

    info = {
        "description": "SharkSpotting - Roboflow splits consolidated into a single raw pool.",
        "url": SOURCE_URL,
        "license": LICENSE["name"],
        "consolidated_from": SOURCE_DIR,
        "source_export_date": source_export_date,
        "source_version": source_version,
        "num_images": len(images),
        "num_annotations": len(annotations),
    }
    out_json = raw_root / "annotations.coco.json"
    out_json.write_text(
        json.dumps(
            {
                "info": info,
                "licenses": [LICENSE],
                "categories": new_categories,
                "images": images,
                "annotations": annotations,
            },
            indent=2,
        )
    )

    _log_summary(images, annotations, new_categories, out_json)


def _verify(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    images_out: Path,
    *,
    expected_images: int = EXPECTED_IMAGES,
    expected_annotations: int = EXPECTED_ANNOTATIONS,
) -> None:
    """Fail loud if the consolidated dataset is internally inconsistent."""
    valid_image_ids = {im["id"] for im in images}
    valid_cat_ids = {c["id"] for c in categories}

    assert len(images) == expected_images, f"expected {expected_images} images, got {len(images)}"
    assert len(annotations) == expected_annotations, (
        f"expected {expected_annotations} annotations, got {len(annotations)}"
    )

    for ann in annotations:
        assert ann["image_id"] in valid_image_ids, f"orphan annotation {ann['id']}"
        assert ann["category_id"] in valid_cat_ids, (
            f"annotation {ann['id']} has bad category {ann['category_id']}"
        )
    for im in images:
        assert (images_out / im["file_name"]).exists(), f"missing image file {im['file_name']}"


def _log_summary(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    out_json: Path,
) -> None:
    name_by_id = {c["id"]: c["name"] for c in categories}
    per_cat: dict[str, int] = {c["name"]: 0 for c in categories}
    for ann in annotations:
        per_cat[name_by_id[ann["category_id"]]] += 1

    log.info("Consolidated %d images, %d annotations", len(images), len(annotations))
    log.info("Annotations per category: %s", per_cat)
    log.info("Wrote %s", out_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty data/raw directory.",
    )
    args = parser.parse_args()
    consolidate(force=args.force)


if __name__ == "__main__":
    main()
