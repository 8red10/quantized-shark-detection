"""ONNX inspection helpers shared by the training stages.

Requires the optional ``onnx`` extra (``qsd-common[onnx]``); imports are done
lazily so that stages without the extra can still import the rest of
``qsd_common``.
"""

from __future__ import annotations

from pathlib import Path


def inspect_onnx(model_path: str | Path) -> dict[str, object]:
    """Return opset + input/output signature for an exported ONNX model.

    Used after export to assert every architecture lands on a quantization-
    compatible opset before it reaches the Jetson.
    """
    import onnx  # lazy: only available with the [onnx] extra

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    graph = model.graph
    return {
        "opset": opsets,
        "inputs": [i.name for i in graph.input],
        "outputs": [o.name for o in graph.output],
    }
