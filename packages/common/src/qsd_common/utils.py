"""Small cross-stage utilities: logging and reproducible seeding."""

from __future__ import annotations

import logging
import os
import random


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python's RNG and ``PYTHONHASHSEED``.

    Framework-specific seeding (numpy/torch) is left to the stages that depend
    on those libraries, so ``common`` stays free of heavy dependencies.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
