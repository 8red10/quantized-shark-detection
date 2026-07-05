"""Stage 3 — quantize (INT8) and benchmark on the Jetson Orin Nano."""

from qsd_common import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("qsd edge: INT8 quantize and benchmark accuracy/latency/power on Jetson")


if __name__ == "__main__":
    main()
