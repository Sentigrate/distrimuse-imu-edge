"""Export to ONNX and apply static int8 post-training quantization.

Why this path exists
--------------------
PyTorch's ``quantize_dynamic`` converts ``Linear`` and ``GRU`` only.
``edge_window_tcn`` carries about 99% of its multiply-accumulates in ``Conv1d``,
so dynamic quantization left the arithmetic almost entirely in float32 — a ~2%
state-dict reduction and no change in effective MAC throughput. That path was
removed from this repository; this module replaces it.

Static quantization does cover convolutions, but it needs **calibration**: a
pass over representative inputs to record the range each activation actually
reaches, so fixed scales and zero-points can be chosen ahead of time. That is
the only thing static quantization asks for beyond dynamic, and it is what makes
convolution support possible.

Why ONNX rather than PyTorch's own quantizer
--------------------------------------------
PyTorch's graph-mode (pt2e) quantization moved out of ``torch`` into
``torchao``, and ``torchao`` does not import on this project's Python 3.14
(``ObserverOrFakeQuantize.__module__`` assignment on a ``typing.Union``, which
3.14 made immutable). The remaining in-torch option is eager-mode quantization,
which would require inserting ``QuantStub``/``DeQuantStub``, replacing the
residual additions in ``_WindowTCNBlock`` with ``FloatFunctional``, and fusing
Conv+BN by hand — invasive edits to model code that the float32 results depend
on.

ONNX Runtime quantizes the exported graph instead, so the model source is
untouched. All three ``edge_window_tcn`` context variants export cleanly.

Deployment caveat
-----------------
ONNX Runtime targets Linux/Android-class hardware, not bare-metal Cortex-M. For
an actual microcontroller the artifact would be TFLite Micro or ExecuTorch. What
this module answers is the question that comes first regardless of runtime: how
much accuracy does int8 cost, and how much smaller does the model get.

Quantization format
-------------------
``QuantFormat.QOperator`` is the default here rather than ``QDQ``. QDQ wraps each
operator in QuantizeLinear/DequantizeLinear pairs and leaves fusion to the
runtime; on a model this small the ~130 extra nodes and their scale tensors cost
more bytes than int8 weights save, so QDQ files come out *larger* than float32.
QOperator emits fused ``QLinearConv``/``QGemm`` nodes directly and does shrink
the file. Measured on ``edge_window_tcn`` at width 0.25 (16,569 parameters):

===============  ========
Format           Size
===============  ========
float32 ONNX     93.1 KiB
QDQ int8         106.5 KiB
QOperator int8   72.0 KiB
===============  ========

Note that 4x weight compression does not mean a 4x file: only 64.7 of those
93.1 KiB are weights, and graph structure does not shrink. The larger prize is
compute — nine ``QLinearConv`` nodes mean the convolutions genuinely execute in
int8, which is the throughput and energy win that dynamic quantization never
delivered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

DEFAULT_OPSET = 17
DEFAULT_INPUT_NAME = "input"
DEFAULT_OUTPUT_NAME = "logits"
# Enough batches to cover the activation ranges without making calibration a
# second training run. ONNX Runtime records running min/max, so coverage of the
# tails matters more than sample count.
DEFAULT_CALIBRATION_BATCHES = 32


def export_onnx(
    model: nn.Module,
    *,
    path: str | Path,
    context_len: int,
    window_size_s: float,
    n_channels: int,
    fs: int = 104,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export ``model`` to a float32 ONNX graph with a dynamic batch axis.

    The batch axis is marked dynamic so the same graph serves both calibration
    (single samples) and evaluation (the datamodule's full batches). Every other
    axis is fixed, matching the deployed input shape.

    Returns:
        The path written.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t = int(round(window_size_s * fs))
    sample = torch.zeros(1, context_len, t, n_channels, dtype=torch.float32)
    torch.onnx.export(
        model.to("cpu").eval(),
        (sample,),
        str(out),
        input_names=[DEFAULT_INPUT_NAME],
        output_names=[DEFAULT_OUTPUT_NAME],
        dynamic_axes={DEFAULT_INPUT_NAME: {0: "batch"}, DEFAULT_OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return out


def _calibration_batches(loader, limit: int) -> Iterator[np.ndarray]:
    for index, (x, _, _) in enumerate(loader):
        if index >= limit:
            return
        yield x.detach().cpu().numpy().astype(np.float32)


def build_calibration_reader(loader, *, limit: int = DEFAULT_CALIBRATION_BATCHES):
    """Wrap a dataloader as an ONNX Runtime ``CalibrationDataReader``.

    Args:
        loader: Must be the **training** split. Calibration reads the input
            distribution, so using validation or test data would leak
            information from the evaluation sets into the deployed model.
        limit: Number of batches to feed.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._batches = _calibration_batches(loader, limit)

        def get_next(self) -> dict[str, np.ndarray] | None:
            batch = next(self._batches, None)
            if batch is None:
                return None
            return {DEFAULT_INPUT_NAME: batch}

    return _Reader()


def quantize_onnx_static(
    fp32_path: str | Path,
    int8_path: str | Path,
    *,
    calibration_reader: Any,
    per_channel: bool = True,
    quant_format: str = "QOperator",
) -> Path:
    """Statically quantize a float32 ONNX graph to int8.

    Weights are signed int8 and activations unsigned int8, the combination ONNX
    Runtime's integer convolution kernels expect. ``per_channel`` gives each
    output channel its own weight scale, which costs a few bytes of metadata and
    usually recovers most of the accuracy that per-tensor scaling loses on
    convolutions.

    Returns:
        The path written.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    out = Path(int8_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(Path(fp32_path).expanduser()),
        str(out),
        calibration_reader,
        quant_format=getattr(QuantFormat, quant_format),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=per_channel,
    )
    return out


class OnnxModule(nn.Module):
    """Adapter presenting an ONNX Runtime session as a callable module.

    ``collect_predictions`` and the report writers only need ``eval()`` and
    ``__call__(tensor) -> tensor``, so wrapping the session this way lets the
    quantized model reuse the entire existing metrics, confusion-matrix,
    per-subject, and plotting path without duplicating any of it.

    Inputs are moved to CPU float32 numpy on the way in and returned as a torch
    tensor on the way out, so callers cannot tell the difference.
    """

    def __init__(self, onnx_path: str | Path) -> None:
        super().__init__()
        import onnxruntime as ort

        self.onnx_path = str(Path(onnx_path).expanduser())
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.onnx_path, options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        array = x.detach().to("cpu").numpy().astype(np.float32)
        logits = self._session.run(None, {self._input_name: array})[0]
        return torch.from_numpy(np.asarray(logits, dtype=np.float32))


def onnx_op_counts(path: str | Path) -> dict[str, int]:
    """Count operator types in an ONNX graph.

    Used to verify that quantization actually reached the convolutions: a
    genuinely int8 graph shows ``QLinearConv`` where the float32 graph showed
    ``Conv``. A compression label cannot be trusted for this, which is the same
    reason the energy model measures its int8 MAC share instead of reading the
    label.
    """
    import onnx

    graph = onnx.load(str(Path(path).expanduser()))
    counts: dict[str, int] = {}
    for node in graph.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def int8_mac_fraction_from_onnx(path: str | Path) -> float:
    """Estimate the share of MAC-bearing operators that run in int8.

    ONNX carries no per-node MAC counts, so this weights the MAC-bearing
    operator types by count rather than by actual arithmetic. For these models
    the convolutions dominate both count and MACs, so the approximation is
    close; it is reported alongside the op counts rather than in place of them.

    Returns 0.0 for a graph with no recognised MAC-bearing operators, so an
    unreadable graph understates efficiency rather than inventing it.
    """
    counts = onnx_op_counts(path)
    float_ops = ("Conv", "Gemm", "MatMul", "ConvTranspose")
    int8_ops = ("QLinearConv", "QGemm", "QLinearMatMul", "ConvInteger", "MatMulInteger")
    n_float = sum(counts.get(op, 0) for op in float_ops)
    n_int8 = sum(counts.get(op, 0) for op in int8_ops)
    total = n_float + n_int8
    if total <= 0:
        return 0.0
    return n_int8 / total
