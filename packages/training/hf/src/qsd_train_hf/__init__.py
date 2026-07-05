"""Stage 2 (HuggingFace) — train a detector via the Transformers Trainer API and export to ONNX."""

from qsd_common import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("qsd train-hf: train with HF Trainer, then export ONNX for quantization")


if __name__ == "__main__":
    main()
