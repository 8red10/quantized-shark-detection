"""Open a project COCO dataset in the FiftyOne app (the ``visualize`` console script).

Pick any of the project's datasets and eyeball its images + ground-truth boxes in the
FiftyOne app: ``just visualize train|val|test|calib|raw|roboflow-{train,valid,test}``.
The ``fiftyone`` dependency group is heavy and not synced by default, so the app import
lives inside ``main()`` and the recipe syncs the group first.

Datasets are DVC-tracked and content-hashed, so this loads them **read-only** (nothing is
persisted or written back into the tracked dirs — see ``.dvcignore``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from qsd_common import data_dir, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's images and (optional) COCO labels live, relative to ``data/``."""

    images: str
    labels: str | None
    dvc_pull: str  # `just pull-*` hint shown if the dataset isn't materialized


# All selectable COCO artifacts. Cleaned datasets keep images under an ``images/`` subdir
# with labels alongside; the roboflow export keeps images + ``_annotations.coco.json`` in
# the same split dir. ``calib`` is image-only (INT8 calibration set, no labels).
DATASETS: dict[str, DatasetSpec] = {
    "raw": DatasetSpec("raw/images", "raw/annotations.coco.json", "just pull-raw"),
    "train": DatasetSpec(
        "processed/train/images", "processed/train/annotations.coco.json",
        "just pull-split train",
    ),
    "val": DatasetSpec(
        "processed/val/images", "processed/val/annotations.coco.json",
        "just pull-split val",
    ),
    "test": DatasetSpec(
        "processed/test/images", "processed/test/annotations.coco.json",
        "just pull-split test",
    ),
    "calib": DatasetSpec("processed/calib/images", None, "just pull-split calib"),
    "roboflow-train": DatasetSpec(
        "roboflow-export/train", "roboflow-export/train/_annotations.coco.json",
        "just pull-rf",
    ),
    "roboflow-valid": DatasetSpec(
        "roboflow-export/valid", "roboflow-export/valid/_annotations.coco.json",
        "just pull-rf",
    ),
    "roboflow-test": DatasetSpec(
        "roboflow-export/test", "roboflow-export/test/_annotations.coco.json",
        "just pull-rf",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASETS),
        help="Which COCO dataset to open in the FiftyOne app.",
    )
    parser.add_argument(
        "--port", type=int, default=5151, help="Port for the FiftyOne app."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="If set, load at most this many samples (fast peek).",
    )
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    images_dir: Path = data_dir() / spec.images
    labels_path: Path | None = data_dir() / spec.labels if spec.labels else None

    if not images_dir.is_dir():
        log.error(
            "%s not found. Pull it first with: %s", images_dir, spec.dvc_pull
        )
        raise SystemExit(1)

    # Heavy import — kept inside main() so the module loads/lints without the group synced.
    import fiftyone as fo
    import fiftyone.types as fot

    if labels_path is not None:
        dataset = fo.Dataset.from_dir(
            dataset_type=fot.COCODetectionDataset,
            data_path=str(images_dir),
            labels_path=str(labels_path),
            label_field="ground_truth",
            max_samples=args.max_samples,
            persistent=False,
        )
    else:
        dataset = fo.Dataset.from_images_dir(
            str(images_dir), max_samples=args.max_samples, persistent=False
        )

    log.info(
        "Loaded '%s' (%d samples) from %s", args.dataset, len(dataset), images_dir
    )
    session = fo.launch_app(dataset, port=args.port)
    log.info("FiftyOne app on http://localhost:%d — Ctrl-C to quit.", args.port)
    session.wait()


if __name__ == "__main__":
    main()
