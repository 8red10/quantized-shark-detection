"""Stage 2 (Roboflow) — train via the Roboflow platform and export to ONNX."""

from qsd_common import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("qsd train-roboflow: train via Roboflow, then export ONNX for quantization")


if __name__ == "__main__":
    main()
