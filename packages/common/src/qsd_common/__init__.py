"""Shared utilities for the Quantized Shark Detection experiment stages."""

from qsd_common.config import load_config
from qsd_common.io import data_dir, manifests_dir, models_dir, repo_root
from qsd_common.notify import (
    TelegramNotifier,
    notify_on_completion,
    send_document,
    send_message,
    send_photo,
)
from qsd_common.utils import get_logger, set_seed

__all__ = [
    "load_config",
    "repo_root",
    "data_dir",
    "models_dir",
    "manifests_dir",
    "get_logger",
    "set_seed",
    "TelegramNotifier",
    "send_message",
    "send_photo",
    "send_document",
    "notify_on_completion",
]

__version__ = "0.1.0"
