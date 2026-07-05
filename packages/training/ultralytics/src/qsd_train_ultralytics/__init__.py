"""Stage 2 (Ultralytics) — train a YOLO model and export to ONNX."""

from qsd_common import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("qsd train-ultralytics: train YOLO, then export ONNX for quantization")


if __name__ == "__main__":
    main()
