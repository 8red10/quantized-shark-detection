"""Config schema for the data-prep stage (loaded from ``configs/data_prep.yaml``)."""

from __future__ import annotations

import math

from pydantic import field_validator

from qsd_common.config import ExperimentConfig
from qsd_common.manifest import SPLITS


class DataPrepConfig(ExperimentConfig):
    """Near-dup grouping, split, and calibration parameters.

    Split ratios come from :class:`ExperimentConfig.splits` (default 80/10/10) so they
    can be changed in the YAML later; a ratio change produces a new manifest and should
    be treated as a new dataset version, not a tweak.
    """

    phash_hash_size: int = 8
    phash_threshold: int = 8
    calib_size: int = 256

    @field_validator("splits")
    @classmethod
    def _validate_splits(cls, splits: dict[str, float]) -> dict[str, float]:
        if set(splits) != set(SPLITS):
            raise ValueError(f"splits keys must be {set(SPLITS)}, got {set(splits)}")
        total = sum(splits.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return splits
