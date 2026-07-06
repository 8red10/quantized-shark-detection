"""Hermetic tests for near-dup grouping (synthetic hashes; one real-pHash anchor test)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from qsd_data_prep.grouping import build_groups, compute_phashes, hamming_matrix


def _hex64(bits_set: int) -> str:
    """A 64-bit hex hash with the lowest ``bits_set`` bits set (Hamming-friendly fixture)."""
    return f"{(1 << bits_set) - 1:016x}"


def test_hamming_matrix_exact() -> None:
    phashes = {0: _hex64(0), 1: _hex64(3), 2: _hex64(64)}
    ids, dist = hamming_matrix(phashes)
    assert ids == [0, 1, 2]
    assert dist[0, 0] == 0
    assert dist[0, 1] == 3
    assert dist[0, 2] == 64
    assert dist[1, 2] == 61
    assert np.array_equal(dist, dist.T)


def test_exact_duplicates_share_a_group() -> None:
    groups = build_groups({0: _hex64(0), 1: _hex64(0), 2: _hex64(64)}, threshold=0)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_distinct_images_stay_separate() -> None:
    groups = build_groups({0: _hex64(0), 1: _hex64(32), 2: _hex64(64)}, threshold=8)
    assert len(set(groups.values())) == 3


def test_transitive_chain_merges_into_one_group() -> None:
    # A~B (6 bits), B~C (6 bits), but A vs C = 12 bits > threshold 8: still one group.
    phashes = {0: _hex64(0), 1: _hex64(6), 2: _hex64(12)}
    ids, dist = hamming_matrix(phashes)
    assert dist[0, 2] == 12
    groups = build_groups(phashes, threshold=8)
    assert groups[0] == groups[1] == groups[2]


def test_group_ids_dense_and_deterministic() -> None:
    phashes = {5: _hex64(64), 2: _hex64(0), 9: _hex64(1), 7: _hex64(32)}
    groups = build_groups(phashes, threshold=4)
    again = build_groups(phashes, threshold=4)
    assert groups == again
    # Dense 0..G-1, ordered by each group's smallest image id: {2,9}=0, 5=1, 7=2.
    assert groups == {2: 0, 9: 0, 5: 1, 7: 2}


def test_raising_threshold_only_merges() -> None:
    phashes = {i: _hex64(i * 5) for i in range(8)}
    low = build_groups(phashes, threshold=4)
    high = build_groups(phashes, threshold=6)
    for a in phashes:
        for b in phashes:
            if low[a] == low[b]:
                assert high[a] == high[b]  # groups never split as threshold rises


def test_compute_phashes_real_images(tmp_path: Path) -> None:
    """Integration anchor: identical files hash identically, distance 0."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "a.jpg")
    Image.fromarray(pixels).save(tmp_path / "b.jpg")

    images = [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"}]
    phashes = compute_phashes(images, tmp_path)
    assert len(phashes[0]) == 16  # 64-bit hex
    assert phashes[0] == phashes[1]
    assert build_groups(phashes, threshold=0)[0] == build_groups(phashes, threshold=0)[1]
