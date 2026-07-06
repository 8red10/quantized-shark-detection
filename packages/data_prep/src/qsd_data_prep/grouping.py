"""Near-duplicate grouping: perceptual hashes + Hamming-distance connected components.

The raw pool contains many near-identical video frames; any split that separates them
leaks. Grouping is transitive on purpose: if frame A ~ B and B ~ C, all three form one
group even when A and C exceed the threshold pairwise — a whole clip must land in a
single split. Over-merging is the conservative direction (it can only make splits more
honest); the pipeline's verify step guards against a runaway mega-group.

Everything here is deterministic (sorted iteration, no RNG), so group ids — and hence
the committed manifest — are byte-stable across runs.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from qsd_common import get_logger

log = get_logger(__name__)


def compute_phashes(
    images: list[dict],
    images_dir: Path,
    *,
    hash_size: int = 8,
) -> dict[int, str]:
    """Perceptual hash per COCO image record, keyed by image id (hex strings)."""
    phashes: dict[int, str] = {}
    for im in tqdm(sorted(images, key=lambda i: i["id"]), desc="phash", unit="img"):
        with Image.open(images_dir / im["file_name"]) as img:
            phashes[im["id"]] = str(imagehash.phash(img, hash_size=hash_size))
    return phashes


def hamming_matrix(phashes: dict[int, str]) -> tuple[list[int], np.ndarray]:
    """All-pairs Hamming distances between hashes.

    Returns ``(image_ids, dist)`` where ``dist[i, j]`` is the bit distance between the
    hashes of ``image_ids[i]`` and ``image_ids[j]``. Hex hashes are unpacked to bit
    vectors; N=4656 yields an NxN uint16 matrix (~43 MB) — trivial to brute-force.
    The XOR broadcast is chunked by rows to cap the intermediate at ~150 MB.
    """
    image_ids = sorted(phashes)
    n = len(image_ids)
    n_bytes = len(phashes[image_ids[0]]) // 2
    packed = np.array(
        [np.frombuffer(bytes.fromhex(phashes[i]), dtype=np.uint8) for i in image_ids]
    )
    assert packed.shape == (n, n_bytes)
    bits = np.unpackbits(packed, axis=1)  # (N, n_bits)
    dist = np.empty((n, n), dtype=np.uint16)
    chunk = 512
    for start in range(0, n, chunk):
        block = bits[start : start + chunk]
        dist[start : start + chunk] = (block[:, None, :] ^ bits[None, :, :]).sum(
            axis=2, dtype=np.uint16
        )
    return image_ids, dist


def build_groups(phashes: dict[int, str], *, threshold: int) -> dict[int, int]:
    """Cluster images into near-dup groups: union-find over pairs with distance <= threshold.

    Returns ``image_id -> group_id`` with group ids re-numbered densely 0..G-1 in
    ascending order of each group's smallest image id (deterministic).
    """
    image_ids, dist = hamming_matrix(phashes)
    n = len(image_ids)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows, cols = np.nonzero(np.triu(dist <= threshold, k=1))
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Canonical root per image (image_ids is sorted, so root index order == min-image-id order).
    roots = [find(i) for i in range(n)]
    group_of_root: dict[int, int] = {}
    groups: dict[int, int] = {}
    for idx, root in enumerate(roots):
        group_id = group_of_root.setdefault(root, len(group_of_root))
        groups[image_ids[idx]] = group_id

    sizes = np.bincount(np.array(list(groups.values())))
    log.info(
        "Grouped %d images into %d near-dup groups (threshold=%d, max group=%d, singletons=%d)",
        n, len(group_of_root), threshold, int(sizes.max()), int((sizes == 1).sum()),
    )
    return groups
