"""Verify materialized splits on disk match the committed manifest.

Loads ``manifests/split_manifest.json`` (which validates it via ``verify_manifest``) and
checks the requested split directory under ``data/processed`` with ``verify_materialized``.
Run after ``dvc pull data/processed/<split>``, before training or benchmarking on the data —
it fails loud (non-zero exit) on any missing/extra image or COCO disagreement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qsd_common import get_logger, load_manifest, verify_materialized
from qsd_common.manifest import CALIB_DIR, SPLITS, default_manifest_path

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="all",
        choices=[*SPLITS, CALIB_DIR, "all"],
        help="Which split to verify (default: all).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the split manifest (default: manifests/split_manifest.json).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Root of the materialized splits (default: data/processed).",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    splits = (*SPLITS, CALIB_DIR) if args.split == "all" else (args.split,)
    for split in splits:
        verify_materialized(split, manifest=manifest, processed_dir=args.processed_dir)
        log.info("split %s: OK", split)
    log.info(
        "verified %d split(s) against %s",
        len(splits),
        args.manifest or default_manifest_path(),
    )
