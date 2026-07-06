"""Stage 1 — near-dup-aware train/val/test splits + INT8 calibration set.

Pipeline (the ``data-prep`` console script):
  1. read the consolidated pool (``data/raw/annotations.coco.json``),
  2. pHash every image and cluster near-duplicates (transitive Hamming components),
  3. assign whole groups to train/val/test, balancing per-class annotation counts,
  4. pick a diverse INT8 calibration subset of train (<=1 image per group),
  5. write ``manifests/split_manifest.json`` (committed to git; byte-identical across
     runs) and materialize ``data/processed/{train,val,test,calib}`` for DVC.

The pHash threshold is derived once via the ``explore-thresholds`` script and pinned
in ``configs/data_prep.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsd_common import (
    ImageEntry,
    SplitManifest,
    data_dir,
    get_logger,
    load_config,
    manifests_dir,
    materialize_splits,
    repo_root,
    set_seed,
    verify_manifest,
)
from qsd_common.manifest import MANIFEST_NAME, SPLITS
from qsd_data_prep.calib import select_calib
from qsd_data_prep.config import DataPrepConfig
from qsd_data_prep.grouping import build_groups, compute_phashes
from qsd_data_prep.split import assign_splits, primary_class, rarity_order

log = get_logger(__name__)

MANIFEST_VERSION = 1


def build_manifest(config: DataPrepConfig, raw_dir: Path) -> SplitManifest:
    """Run grouping + splitting + calib selection over ``raw_dir`` and assemble the manifest."""
    coco = json.loads((raw_dir / "annotations.coco.json").read_text())
    images, annotations, categories = coco["images"], coco["annotations"], coco["categories"]

    phashes = compute_phashes(images, raw_dir / "images", hash_size=config.phash_hash_size)
    groups = build_groups(phashes, threshold=config.phash_threshold)
    assignment = assign_splits(images, annotations, categories, groups, config.splits)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    calib = select_calib(
        assignment, groups, anns_by_image, categories, calib_size=config.calib_size
    )

    rarity = rarity_order(annotations, categories)
    name_by_id = {c["id"]: c["name"] for c in categories}
    entries = [
        ImageEntry(
            image_id=im["id"],
            file_name=im["file_name"],
            phash=phashes[im["id"]],
            group_id=groups[im["id"]],
            primary_class=primary_class(im["id"], anns_by_image, rarity, name_by_id),
            split=assignment[im["id"]],
            is_calib=im["id"] in calib,
        )
        for im in sorted(images, key=lambda i: i["id"])
    ]

    import imagehash
    import PIL

    group_sizes: dict[int, int] = {}
    for gid in groups.values():
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
    anns_per_split_class = {
        s: dict.fromkeys((name_by_id[c["id"]] for c in categories), 0) for s in SPLITS
    }
    for ann in annotations:
        split = assignment[ann["image_id"]]
        anns_per_split_class[split][name_by_id[ann["category_id"]]] += 1

    meta = {
        "manifest_version": MANIFEST_VERSION,
        "config_name": config.name,
        "source": "data/raw/annotations.coco.json",
        "source_num_images": len(images),
        "source_num_annotations": len(annotations),
        "phash": {
            "hash_size": config.phash_hash_size,
            "threshold": config.phash_threshold,
            # Hash values depend on these libraries; a version bump explains a manifest diff.
            "imagehash_version": imagehash.__version__,
            "pillow_version": PIL.__version__,
        },
        "seed": config.seed,
        "splits": config.splits,
        "calib_size": config.calib_size,
    }
    summary = {
        "num_groups": len(group_sizes),
        "max_group_size": max(group_sizes.values()),
        "num_calib": len(calib),
        "images_per_split": {
            s: sum(1 for e in entries if e.split == s) for s in SPLITS
        },
        "annotations_per_split_per_class": anns_per_split_class,
    }
    return SplitManifest(meta=meta, summary=summary, images=entries)


def run(config: DataPrepConfig, *, force: bool = False, materialize: bool = True) -> Path:
    set_seed(config.seed)
    raw_dir = data_dir() / "raw"
    manifest_path = manifests_dir() / MANIFEST_NAME

    if manifest_path.exists() and not force:
        raise SystemExit(
            f"{manifest_path} already exists. Re-run with --force to rebuild it (output is "
            f"deterministic, so a --force rebuild is byte-identical unless config or data "
            f"changed)."
        )

    manifest = build_manifest(config, raw_dir)
    verify_manifest(manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Wrote %s", manifest_path)
    log.info("Summary: %s", manifest.summary)

    if materialize:
        materialize_splits(manifest, raw_dir=raw_dir, force=force)
        log.info("Materialized splits under %s", data_dir() / "processed")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the data-prep YAML config (default: configs/data_prep.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest and data/processed tree.",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Only write the manifest; skip building data/processed.",
    )
    args = parser.parse_args()

    config_path = args.config or repo_root() / "configs" / "data_prep.yaml"
    config = load_config(config_path, DataPrepConfig)
    log.info("Loaded config %s: %s", config_path, config.model_dump())
    run(config, force=args.force, materialize=not args.no_materialize)


if __name__ == "__main__":
    main()
