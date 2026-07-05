"""Shared experiment-config schema and YAML loader.

Each stage extends :class:`ExperimentConfig` (or loads a subset) so that split
names, seeds, and artifact locations stay consistent across machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ExperimentConfig(BaseModel):
    """Baseline config fields shared by every stage."""

    name: str = Field(description="Experiment / run name.")
    seed: int = 42
    splits: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )


def load_config(path: str | Path, model: type[T] = ExperimentConfig) -> T:
    """Load a YAML config file and validate it against ``model``."""
    data = yaml.safe_load(Path(path).read_text())
    return model.model_validate(data)
