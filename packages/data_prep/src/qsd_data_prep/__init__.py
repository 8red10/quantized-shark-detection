"""Stage 1 — data preparation (splits + calibration set)."""

from qsd_common import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("qsd data-prep: build near-dup-aware splits and INT8 calibration set")


if __name__ == "__main__":
    main()
