"""One-time pHash threshold derivation aid (the ``explore-thresholds`` console script).

For each candidate threshold, reports how the near-dup grouping would shake out
(group count, singletons, max group size) so a runaway chain-merge is obvious. With
``--montages N`` it also writes side-by-side pair images sampled per Hamming-distance
band under ``data/scratch/threshold-explore/`` (gitignored) — eyeball those to pick the
largest threshold whose pairs still look like true near-duplicates, then pin it as
``phash_threshold`` in ``configs/data_prep.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from qsd_common import data_dir, get_logger
from qsd_data_prep.grouping import build_groups, compute_phashes

log = get_logger(__name__)

MONTAGE_HEIGHT = 360


def _pair_montage(images_dir: Path, name_a: str, name_b: str) -> Image.Image:
    halves = []
    for name in (name_a, name_b):
        with Image.open(images_dir / name) as img:
            scale = MONTAGE_HEIGHT / img.height
            halves.append(img.resize((int(img.width * scale), MONTAGE_HEIGHT)).convert("RGB"))
    canvas = Image.new("RGB", (halves[0].width + halves[1].width, MONTAGE_HEIGHT))
    canvas.paste(halves[0], (0, 0))
    canvas.paste(halves[1], (halves[0].width, 0))
    return canvas


def _write_montages(
    images: list[dict],
    images_dir: Path,
    image_ids: list[int],
    dist: np.ndarray,
    bands: list[int],
    per_band: int,
    out_dir: Path,
) -> None:
    """Sample ``per_band`` pairs at each exact Hamming distance and save side-by-sides."""
    name_by_id = {im["id"]: im["file_name"] for im in images}
    rng = np.random.default_rng(0)  # sampling is cosmetic; still seeded for repeatability
    for band in bands:
        rows, cols = np.nonzero(np.triu(dist == band, k=1))
        if not len(rows):
            log.info("distance %d: no pairs", band)
            continue
        band_dir = out_dir / f"distance_{band:02d}"
        band_dir.mkdir(parents=True, exist_ok=True)
        picks = rng.choice(len(rows), size=min(per_band, len(rows)), replace=False)
        for k in picks:
            a, b = image_ids[rows[k]], image_ids[cols[k]]
            montage = _pair_montage(images_dir, name_by_id[a], name_by_id[b])
            montage.save(band_dir / f"pair_{a:06d}_{b:06d}.jpg", quality=85)
        log.info("distance %d: %d pairs total, wrote %d montages to %s",
                 band, len(rows), len(picks), band_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14],
        help="Candidate Hamming thresholds to report grouping stats for.",
    )
    parser.add_argument(
        "--hash-size", type=int, default=8, help="pHash hash size (8 -> 64-bit)."
    )
    parser.add_argument(
        "--montages", type=int, default=0,
        help="If >0, write this many sample pair montages per Hamming-distance band.",
    )
    args = parser.parse_args()

    raw_dir = data_dir() / "raw"
    coco = json.loads((raw_dir / "annotations.coco.json").read_text())
    images = coco["images"]

    phashes = compute_phashes(images, raw_dir / "images", hash_size=args.hash_size)

    from qsd_data_prep.grouping import hamming_matrix

    image_ids, dist = hamming_matrix(phashes)
    log.info("%-9s %8s %10s %10s", "threshold", "groups", "singletons", "max group")
    for threshold in sorted(args.thresholds):
        groups = build_groups(phashes, threshold=threshold)
        sizes = np.bincount(np.array(list(groups.values())))
        log.info("%-9d %8d %10d %10d",
                 threshold, len(sizes), int((sizes == 1).sum()), int(sizes.max()))

    if args.montages > 0:
        bands = sorted({b for t in args.thresholds for b in (t - 1, t, t + 1) if b >= 1})
        out_dir = data_dir() / "scratch" / "threshold-explore"
        _write_montages(
            images, raw_dir / "images", image_ids, dist, bands, args.montages, out_dir
        )
        log.info("Review montages under %s, then set phash_threshold in configs/data_prep.yaml",
                 out_dir)


if __name__ == "__main__":
    main()
