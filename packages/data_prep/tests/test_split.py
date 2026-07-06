"""Tests for group-aware split assignment and calibration selection (synthetic data)."""

from __future__ import annotations

import pytest

from qsd_data_prep.calib import select_calib
from qsd_data_prep.split import assign_splits, primary_class, rarity_order

CATEGORIES = [{"id": 0, "name": "boat"}, {"id": 1, "name": "shark"}, {"id": 2, "name": "person"}]
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _build_dataset(
    n_groups: int = 60,
    group_size: int = 3,
    boat_every: int = 5,
    background_every: int = 6,
):
    """Synthetic pool: ``n_groups`` near-dup groups of ``group_size`` images each.

    Every ``boat_every``-th group carries rare-class (boat) annotations; every
    ``background_every``-th group is background-only. Others get shark + person.
    """
    images, annotations, groups = [], [], {}
    ann_id = 1
    image_id = 0
    for gid in range(n_groups):
        for _ in range(group_size):
            images.append({"id": image_id, "file_name": f"img_{image_id:04d}.jpg"})
            groups[image_id] = gid
            if gid % background_every == 0:
                pass  # background-only group
            elif gid % boat_every == 0:
                for cid in (0, 2):
                    annotations.append(
                        {"id": ann_id, "image_id": image_id, "category_id": cid,
                         "bbox": [0, 0, 1, 1], "area": 1}
                    )
                    ann_id += 1
            else:
                for cid in (1, 2, 2):  # person twice: keeps it the common class
                    annotations.append(
                        {"id": ann_id, "image_id": image_id, "category_id": cid,
                         "bbox": [0, 0, 1, 1], "area": 1}
                    )
                    ann_id += 1
            image_id += 1
    return images, annotations, groups


def test_no_group_straddles_splits() -> None:
    images, annotations, groups = _build_dataset()
    assignment = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    split_by_group: dict[int, str] = {}
    for image_id, split in assignment.items():
        gid = groups[image_id]
        assert split_by_group.setdefault(gid, split) == split


def test_every_class_in_every_split_and_ratios() -> None:
    images, annotations, groups = _build_dataset()
    assignment = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)

    per_class_split = {c["id"]: {"train": 0, "val": 0, "test": 0} for c in CATEGORIES}
    for ann in annotations:
        per_class_split[ann["category_id"]][assignment[ann["image_id"]]] += 1
    for cid, counts in per_class_split.items():
        for split, n in counts.items():
            assert n > 0, f"class {cid} missing from {split}"

    n = len(images)
    for split, ratio in RATIOS.items():
        actual = sum(1 for s in assignment.values() if s == split) / n
        assert abs(actual - ratio) <= 0.05


def test_background_only_groups_distributed() -> None:
    images, annotations, groups = _build_dataset()
    assignment = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    annotated = {a["image_id"] for a in annotations}
    background_splits = {assignment[im["id"]] for im in images if im["id"] not in annotated}
    assert background_splits == {"train", "val", "test"}


def test_assignment_deterministic() -> None:
    images, annotations, groups = _build_dataset()
    first = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    second = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    assert first == second


def test_custom_ratios_are_respected() -> None:
    images, annotations, groups = _build_dataset(n_groups=100, group_size=2)
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    assignment = assign_splits(images, annotations, CATEGORIES, groups, ratios)
    n = len(images)
    for split, ratio in ratios.items():
        actual = sum(1 for s in assignment.values() if s == split) / n
        assert abs(actual - ratio) <= 0.05


def test_mega_group_fails_loud() -> None:
    images, annotations, groups = _build_dataset()
    for image_id in groups:  # chain-merge half the pool into group 0
        if image_id < len(images) // 2:
            groups[image_id] = 0
    with pytest.raises(AssertionError, match="chain-merged"):
        assign_splits(images, annotations, CATEGORIES, groups, RATIOS)


def test_rarity_order_and_primary_class() -> None:
    _, annotations, _ = _build_dataset()
    rarity = rarity_order(annotations, CATEGORIES)
    assert rarity[0] == 0  # boat rarest
    assert rarity[-1] == 2  # person most common

    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    name_by_id = {c["id"]: c["name"] for c in CATEGORIES}

    boat_img = next(a["image_id"] for a in annotations if a["category_id"] == 0)
    assert primary_class(boat_img, anns_by_image, rarity, name_by_id) == "boat"
    background_img = next(
        i for i in range(len(anns_by_image) + 10) if i not in anns_by_image
    )
    assert primary_class(background_img, anns_by_image, rarity, name_by_id) is None


def test_calib_selection_invariants() -> None:
    images, annotations, groups = _build_dataset()
    assignment = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    calib = select_calib(assignment, groups, anns_by_image, CATEGORIES, calib_size=20)
    assert len(calib) == 20
    assert all(assignment[i] == "train" for i in calib)
    calib_groups = [groups[i] for i in calib]
    assert len(calib_groups) == len(set(calib_groups))  # <=1 per near-dup group
    # Rare class and background both represented (round-robin coverage).
    assert any(any(a["category_id"] == 0 for a in anns_by_image.get(i, [])) for i in calib)
    assert any(i not in anns_by_image for i in calib)

    again = select_calib(assignment, groups, anns_by_image, CATEGORIES, calib_size=20)
    assert calib == again  # deterministic


def test_calib_capped_by_eligible_groups() -> None:
    # Small pool (but with enough rare-class groups to cover all three splits).
    images, annotations, groups = _build_dataset(n_groups=20)
    assignment = assign_splits(images, annotations, CATEGORIES, groups, RATIOS)
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    n_train_groups = len({groups[i] for i, s in assignment.items() if s == "train"})
    calib = select_calib(assignment, groups, anns_by_image, CATEGORIES, calib_size=500)
    assert len(calib) == n_train_groups
