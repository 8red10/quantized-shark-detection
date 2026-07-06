"""Group-aware stratified split assignment.

Whole near-dup groups are assigned to a single split (no leakage by construction),
greedily choosing the split that minimizes a squared relative-deficit cost over
per-class annotation counts plus an image-count term. Relative fractions self-weight
rare classes (boat's 248 annotations move its fraction ~17x more per annotation than
person's 4292), so rare classes get balanced first-class treatment without explicit
weights. Groups are processed rare-class-first while every split still has room.

Deterministic by construction: sorted iteration and fixed tie-breaking, no RNG.
"""

from __future__ import annotations

from qsd_common import get_logger

log = get_logger(__name__)

SPLIT_ORDER = ("train", "val", "test")

# Weight of the image-count term relative to the per-class annotation terms. Small, so
# annotation balance dominates; it is the sole driver for background-only groups.
IMAGE_TERM_WEIGHT = 0.25

# A single near-dup group larger than this fraction of the pool means the threshold
# chain-merged a huge portion of the dataset — fail loud instead of producing a
# degenerate, unstratifiable split.
MAX_GROUP_FRACTION = 0.25


def rarity_order(annotations: list[dict], categories: list[dict]) -> list[int]:
    """Category ids sorted rarest-first by global annotation count (ties by id)."""
    counts = {c["id"]: 0 for c in categories}
    for ann in annotations:
        counts[ann["category_id"]] += 1
    return sorted(counts, key=lambda cid: (counts[cid], cid))


def primary_class(
    image_id: int,
    anns_by_image: dict[int, list[dict]],
    rarity: list[int],
    name_by_id: dict[int, str],
) -> str | None:
    """Name of the rarest class present in the image (audit column); None if unannotated."""
    present = {a["category_id"] for a in anns_by_image.get(image_id, [])}
    for cid in rarity:
        if cid in present:
            return name_by_id[cid]
    return None


def assign_splits(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    groups: dict[int, int],
    ratios: dict[str, float],
) -> dict[int, str]:
    """Assign every image's near-dup group to one split; returns ``image_id -> split``."""
    cat_ids = [c["id"] for c in categories]
    rarity = rarity_order(annotations, categories)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    group_class_counts: dict[int, dict[int, int]] = {}
    group_image_counts: dict[int, int] = {}
    for im in images:
        gid = groups[im["id"]]
        group_image_counts[gid] = group_image_counts.get(gid, 0) + 1
        counts = group_class_counts.setdefault(gid, dict.fromkeys(cat_ids, 0))
        for ann in anns_by_image.get(im["id"], []):
            counts[ann["category_id"]] += 1

    total_images = len(images)
    total_per_class = {cid: 0 for cid in cat_ids}
    for counts in group_class_counts.values():
        for cid, n in counts.items():
            total_per_class[cid] += n

    def order_key(gid: int) -> tuple:
        counts = group_class_counts[gid]
        total_anns = sum(counts.values())
        # Rank of the rarest class present (len(rarity) = background-only, placed last).
        rank = next((r for r, cid in enumerate(rarity) if counts[cid] > 0), len(rarity))
        rare_count = counts[rarity[rank]] if rank < len(rarity) else 0
        return (rank, -rare_count, -total_anns, gid)

    assigned_class: dict[str, dict[int, int]] = {
        s: dict.fromkeys(cat_ids, 0) for s in SPLIT_ORDER
    }
    assigned_images = dict.fromkeys(SPLIT_ORDER, 0)
    group_split: dict[int, str] = {}

    def place(gid: int, split: str) -> None:
        group_split[gid] = split
        assigned_images[split] += group_image_counts[gid]
        for cid in cat_ids:
            assigned_class[split][cid] += group_class_counts[gid][cid]

    # Coverage seeding: the squared-deficit objective alone can starve a split of a class
    # whose annotations live in only a handful of near-dup groups (its optimum may put all
    # of them in train). Guarantee every class one group per split up front, rarest class
    # first and smallest splits first, using the groups that distort the balance least.
    for cid in rarity:
        if total_per_class[cid] == 0:
            continue
        for split in sorted(SPLIT_ORDER, key=lambda s: (ratios[s], SPLIT_ORDER.index(s))):
            covered = any(
                group_class_counts[g][cid] > 0
                for g, s in group_split.items()
                if s == split
            )
            if covered:
                continue
            candidates = [
                g
                for g, counts in group_class_counts.items()
                if g not in group_split and counts[cid] > 0
            ]
            if not candidates:
                break  # not enough groups to cover every split; _verify_splits will decide
            place(
                min(
                    candidates,
                    key=lambda g: (
                        group_class_counts[g][cid],
                        sum(group_class_counts[g].values()),
                        g,
                    ),
                ),
                split,
            )

    for gid in sorted(group_class_counts, key=order_key):
        if gid in group_split:
            continue  # placed by coverage seeding
        counts = group_class_counts[gid]

        def cost_delta(split: str, counts: dict[int, int] = counts, gid: int = gid) -> float:
            """Change in the global squared-deficit objective if the group joins ``split``.

            Comparing post-assignment deltas (not absolute deficits) is what makes the
            greedy sound: an absolute cost would penalize train for every class it has
            not filled yet and shunt early groups into the small splits.
            """
            delta = 0.0
            for cid in cat_ids:
                if total_per_class[cid] == 0 or counts[cid] == 0:
                    continue
                before = assigned_class[split][cid] / total_per_class[cid]
                after = (assigned_class[split][cid] + counts[cid]) / total_per_class[cid]
                delta += (after - ratios[split]) ** 2 - (before - ratios[split]) ** 2
            img_before = assigned_images[split] / total_images
            img_after = (assigned_images[split] + group_image_counts[gid]) / total_images
            delta += IMAGE_TERM_WEIGHT * (
                (img_after - ratios[split]) ** 2 - (img_before - ratios[split]) ** 2
            )
            return delta

        best = min(SPLIT_ORDER, key=lambda s: (cost_delta(s), SPLIT_ORDER.index(s)))
        place(gid, best)

    assignment = {im["id"]: group_split[groups[im["id"]]] for im in images}
    _verify_splits(assignment, groups, annotations, categories, ratios)

    name_by_id = {c["id"]: c["name"] for c in categories}
    for split in SPLIT_ORDER:
        per_class = {name_by_id[cid]: assigned_class[split][cid] for cid in cat_ids}
        log.info(
            "%s: %d images (%.1f%%), annotations per class: %s",
            split, assigned_images[split],
            100 * assigned_images[split] / total_images, per_class,
        )
    return assignment


def _verify_splits(
    assignment: dict[int, str],
    groups: dict[int, int],
    annotations: list[dict],
    categories: list[dict],
    ratios: dict[str, float],
    *,
    ratio_tolerance: float = 0.05,
    max_group_fraction: float = MAX_GROUP_FRACTION,
) -> None:
    """Fail loud on leakage, missing class coverage, or badly skewed ratios."""
    split_by_group: dict[int, str] = {}
    group_sizes: dict[int, int] = {}
    for image_id, split in assignment.items():
        gid = groups[image_id]
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
        prior = split_by_group.setdefault(gid, split)
        assert prior == split, f"group {gid} straddles splits ({prior} vs {split}) — leakage"

    biggest = max(group_sizes, key=lambda g: group_sizes[g])
    frac = group_sizes[biggest] / len(assignment)
    assert frac < max_group_fraction, (
        f"near-dup group {biggest} holds {frac:.0%} of all images — threshold chain-merged "
        f"too much; lower phash_threshold (see explore-thresholds)"
    )

    class_split_counts = {c["id"]: dict.fromkeys(SPLIT_ORDER, 0) for c in categories}
    for ann in annotations:
        class_split_counts[ann["category_id"]][assignment[ann["image_id"]]] += 1
    for cat in categories:
        for split in SPLIT_ORDER:
            assert class_split_counts[cat["id"]][split] > 0, (
                f"class {cat['name']!r} has no annotations in split {split!r}"
            )

    n = len(assignment)
    for split in SPLIT_ORDER:
        actual = sum(1 for s in assignment.values() if s == split) / n
        assert abs(actual - ratios[split]) <= ratio_tolerance, (
            f"split {split!r} has {actual:.1%} of images, target {ratios[split]:.1%} "
            f"(±{ratio_tolerance:.0%})"
        )
