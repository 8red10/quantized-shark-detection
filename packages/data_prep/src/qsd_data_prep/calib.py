"""INT8 calibration-set selection: diverse train images for TensorRT PTQ on the edge.

At most one image per near-dup group (near-identical frames add nothing to calibration
histograms), chosen round-robin over classes rarest-first so every class's activation
statistics are represented. Deterministic: sorted candidates, fixed tie-breaking.
"""

from __future__ import annotations

from qsd_data_prep.split import rarity_order


def select_calib(
    assignment: dict[int, str],
    groups: dict[int, int],
    anns_by_image: dict[int, list[dict]],
    categories: list[dict],
    *,
    calib_size: int = 256,
) -> set[int]:
    """Pick up to ``calib_size`` train image ids, at most one per near-dup group."""
    annotations = [a for anns in anns_by_image.values() for a in anns]
    rarity = rarity_order(annotations, categories)

    # One representative per train group: the image with the most annotations (tie: lowest id).
    rep_of_group: dict[int, int] = {}
    for image_id in sorted(assignment):
        if assignment[image_id] != "train":
            continue
        gid = groups[image_id]
        best = rep_of_group.get(gid)
        if best is None or len(anns_by_image.get(image_id, [])) > len(anns_by_image.get(best, [])):
            rep_of_group[gid] = image_id

    def class_count(image_id: int, cid: int) -> int:
        return sum(1 for a in anns_by_image.get(image_id, []) if a["category_id"] == cid)

    # Per-class candidate queues (reps having that class, best-first) + background queue.
    queues: list[list[int]] = []
    for cid in rarity:
        members = [r for r in rep_of_group.values() if class_count(r, cid) > 0]
        members.sort(key=lambda r: (-class_count(r, cid), r))
        queues.append(members)
    background = sorted(r for r in rep_of_group.values() if not anns_by_image.get(r))
    queues.append(background)

    selected: set[int] = set()
    positions = [0] * len(queues)
    while len(selected) < calib_size:
        progressed = False
        for qi, queue in enumerate(queues):
            if len(selected) >= calib_size:
                break
            while positions[qi] < len(queue) and queue[positions[qi]] in selected:
                positions[qi] += 1
            if positions[qi] < len(queue):
                selected.add(queue[positions[qi]])
                positions[qi] += 1
                progressed = True
        if not progressed:
            break  # fewer eligible groups than calib_size
    return selected
