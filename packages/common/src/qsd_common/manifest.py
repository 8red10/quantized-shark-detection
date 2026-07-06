"""Split-manifest schema, loader, and split materialization/verification helpers.

The data-prep stage writes ``manifests/split_manifest.json`` — the single source of
truth mapping every image in ``data/raw`` to a near-duplicate group, a train/val/test
split, and (for a train subset) the INT8 calibration set. Later stages consume it via:

  * :func:`load_manifest` — parse + validate the committed manifest,
  * :func:`materialize_splits` — rebuild ``data/processed/{train,val,test,calib}``
    from ``data/raw`` + the manifest (deterministic bytes, DVC-friendly),
  * :func:`verify_materialized` — after ``dvc pull``, check a split directory on disk
    matches the manifest exactly (fail loud before training/benchmarking on it).

Stdlib + pydantic only: the Jetson edge stage (Python 3.10, no numpy/pillow in
``qsd-common``) must be able to import this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from qsd_common.io import clear_dir, data_dir, manifests_dir

MANIFEST_NAME = "split_manifest.json"
SPLITS = ("train", "val", "test")
CALIB_DIR = "calib"


class ImageEntry(BaseModel):
    """One image's row in the split manifest."""

    image_id: int
    file_name: str
    phash: str
    group_id: int
    primary_class: str | None
    split: str
    is_calib: bool


class SplitManifest(BaseModel):
    """The full manifest: provenance meta, summary stats, and one entry per image."""

    meta: dict
    summary: dict
    images: list[ImageEntry]

    def by_split(self, split: str) -> list[ImageEntry]:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        return [im for im in self.images if im.split == split]

    def calib_images(self) -> list[ImageEntry]:
        return [im for im in self.images if im.is_calib]


def default_manifest_path() -> Path:
    return manifests_dir() / MANIFEST_NAME


def load_manifest(path: Path | None = None) -> SplitManifest:
    """Load and validate the split manifest (defaults to ``manifests/split_manifest.json``)."""
    path = path or default_manifest_path()
    manifest = SplitManifest.model_validate_json(path.read_text())
    verify_manifest(manifest)
    return manifest


def verify_manifest(manifest: SplitManifest) -> None:
    """Fail loud if the manifest is internally inconsistent."""
    ids = [im.image_id for im in manifest.images]
    assert len(ids) == len(set(ids)), "duplicate image_id in manifest"
    names = [im.file_name for im in manifest.images]
    assert len(names) == len(set(names)), "duplicate file_name in manifest"

    split_by_group: dict[int, str] = {}
    for im in manifest.images:
        assert im.split in SPLITS, f"image {im.image_id} has unknown split {im.split!r}"
        prior = split_by_group.setdefault(im.group_id, im.split)
        assert prior == im.split, (
            f"near-dup group {im.group_id} straddles splits ({prior} vs {im.split}) — leakage"
        )
        if im.is_calib:
            assert im.split == "train", f"calib image {im.image_id} is in {im.split}, not train"

    summary_counts = manifest.summary.get("images_per_split")
    if summary_counts is not None:
        actual = {s: sum(1 for im in manifest.images if im.split == s) for s in SPLITS}
        assert summary_counts == actual, (
            f"summary images_per_split {summary_counts} != actual {actual}"
        )


def materialize_splits(
    manifest: SplitManifest | None = None,
    *,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Build ``data/processed/{train,val,test,calib}`` from ``data/raw`` + the manifest.

    Output is deterministic (sorted ids, no timestamps): identical image bytes are copied
    (DVC's content-addressable cache dedups them against ``data/raw`` in the remote) and
    per-split COCO JSONs are byte-identical across runs, so re-running never churns DVC.
    """
    import shutil

    manifest = manifest or load_manifest()
    raw_dir = raw_dir or data_dir() / "raw"
    out_dir = out_dir or data_dir() / "processed"

    coco = json.loads((raw_dir / "annotations.coco.json").read_text())
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only the four split dirs are ours to (re)build; leave DVC's files (.gitignore,
    # *.dvc pointers) in out_dir untouched so a --force rebuild stays dvc-add-able.
    targets = [out_dir / s for s in (*SPLITS, CALIB_DIR)]
    existing = [t for t in targets if t.exists() and any(t.iterdir())]
    if existing:
        if not force:
            raise SystemExit(
                f"{existing[0]} already exists and is not empty. Re-run with force=True/--force "
                f"(output is deterministic, so a rebuild is byte-identical and DVC-safe)."
            )
        for target in existing:
            clear_dir(target)

    for split in SPLITS:
        entries = sorted(manifest.by_split(split), key=lambda im: im.image_id)
        split_dir = out_dir / split
        images_out = split_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        split_images: list[dict] = []
        split_anns: list[dict] = []
        for entry in entries:
            src = images_by_id.get(entry.image_id)
            assert src is not None, f"manifest image {entry.image_id} missing from raw COCO"
            assert src["file_name"] == entry.file_name, (
                f"manifest file_name {entry.file_name} != raw {src['file_name']} "
                f"for image {entry.image_id}"
            )
            shutil.copy2(raw_dir / "images" / entry.file_name, images_out / entry.file_name)
            split_images.append(src)
            split_anns.extend(anns_by_image.get(entry.image_id, []))
        split_anns.sort(key=lambda a: a["id"])

        info = dict(coco.get("info", {}))
        info["split"] = split
        info["split_manifest_meta"] = manifest.meta
        info["num_images"] = len(split_images)
        info["num_annotations"] = len(split_anns)
        (split_dir / "annotations.coco.json").write_text(
            json.dumps(
                {
                    "info": info,
                    "licenses": coco.get("licenses", []),
                    "categories": coco["categories"],
                    "images": split_images,
                    "annotations": split_anns,
                },
                indent=2,
            )
        )

    calib_out = out_dir / CALIB_DIR / "images"
    calib_out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(manifest.calib_images(), key=lambda im: im.image_id):
        shutil.copy2(raw_dir / "images" / entry.file_name, calib_out / entry.file_name)

    # Prune Finder droppings so the on-disk tree matches the deterministic file set exactly
    # (same self-healing guard as consolidate: keeps rebuilds byte-identical and DVC-safe).
    for junk in out_dir.rglob(".DS_Store"):
        junk.unlink(missing_ok=True)


def verify_materialized(
    split: str,
    *,
    manifest: SplitManifest | None = None,
    processed_dir: Path | None = None,
) -> None:
    """Fail loud unless the on-disk split directory matches the manifest exactly.

    Meant for training/edge stages right after ``dvc pull data/processed/<split>``:
    checks the image file set (no missing, no extras) and, for train/val/test, that the
    per-split COCO JSON agrees with the manifest. ``split`` may also be ``"calib"``.
    """
    manifest = manifest or load_manifest()
    processed_dir = processed_dir or data_dir() / "processed"
    split_dir = processed_dir / split

    if split == CALIB_DIR:
        expected = {im.file_name for im in manifest.calib_images()}
    else:
        expected = {im.file_name for im in manifest.by_split(split)}

    images_dir = split_dir / "images"
    assert images_dir.is_dir(), f"missing {images_dir} — did you run dvc pull / materialize?"
    on_disk = {p.name for p in images_dir.iterdir() if p.name != ".DS_Store"}
    missing = expected - on_disk
    extra = on_disk - expected
    assert not missing, f"{split}: {len(missing)} image(s) missing, e.g. {sorted(missing)[:3]}"
    assert not extra, f"{split}: {len(extra)} unexpected file(s), e.g. {sorted(extra)[:3]}"

    if split != CALIB_DIR:
        coco = json.loads((split_dir / "annotations.coco.json").read_text())
        coco_names = {im["file_name"] for im in coco["images"]}
        assert coco_names == expected, f"{split}: COCO json image set disagrees with manifest"
        image_ids = {im["id"] for im in coco["images"]}
        for ann in coco["annotations"]:
            assert ann["image_id"] in image_ids, f"{split}: orphan annotation {ann['id']}"
