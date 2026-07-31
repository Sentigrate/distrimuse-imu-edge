from __future__ import annotations

import numpy as np
import pytest
import torch

from distrimuse_imu_edge.compression.onnx_int8 import (
    OnnxModule,
    build_calibration_reader,
    export_onnx,
    int8_mac_fraction_from_onnx,
    onnx_op_counts,
    quantize_onnx_static,
)
from distrimuse_imu_edge.models import build_model

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

CONTEXT_LEN = 8
WINDOW_S = 0.2
FS = 100
N_CHANNELS = 6
T = int(WINDOW_S * FS)


def _model() -> torch.nn.Module:
    return build_model(
        "edge_window_tcn",
        n_classes=9,
        input_channels=N_CHANNELS,
        width_mult=0.25,
        current_index=CONTEXT_LEN - 1,
    ).eval()


def _fake_loader(n_batches: int = 4, batch_size: int = 2):
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        x = torch.from_numpy(
            rng.standard_normal((batch_size, CONTEXT_LEN, T, N_CHANNELS), dtype=np.float32)
        )
        y = torch.zeros(batch_size, dtype=torch.long)
        yield x, y, {}


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[torch.nn.Module, object]:
    model = _model()
    path = tmp_path_factory.mktemp("onnx") / "fp32.onnx"
    export_onnx(
        model,
        path=path,
        context_len=CONTEXT_LEN,
        window_size_s=WINDOW_S,
        n_channels=N_CHANNELS,
        fs=FS,
    )
    return model, path


def test_export_matches_pytorch_outputs(exported) -> None:
    model, path = exported
    session = OnnxModule(path)
    x = torch.from_numpy(
        np.random.default_rng(1).standard_normal((3, CONTEXT_LEN, T, N_CHANNELS), dtype=np.float32)
    )

    with torch.no_grad():
        reference = model(x)

    assert torch.allclose(session(x), reference, atol=1e-4)


def test_export_accepts_a_dynamic_batch_axis(exported) -> None:
    """One graph must serve batch-1 calibration and batched evaluation."""
    _, path = exported
    session = OnnxModule(path)

    for batch in (1, 5):
        x = torch.zeros(batch, CONTEXT_LEN, T, N_CHANNELS)
        assert session(x).shape == (batch, 9)


def test_float32_graph_has_unquantized_convolutions(exported) -> None:
    _, path = exported
    counts = onnx_op_counts(path)

    assert counts.get("Conv", 0) > 0
    assert counts.get("QLinearConv", 0) == 0
    assert int8_mac_fraction_from_onnx(path) == 0.0


def test_static_quantization_converts_the_convolutions(exported, tmp_path) -> None:
    """The defect dynamic quantization had: convolutions must actually go int8."""
    _, fp32 = exported
    int8 = tmp_path / "int8.onnx"
    quantize_onnx_static(
        fp32,
        int8,
        calibration_reader=build_calibration_reader(_fake_loader(), limit=4),
    )
    counts = onnx_op_counts(int8)

    assert counts.get("QLinearConv", 0) > 0
    assert counts.get("Conv", 0) == 0
    assert int8_mac_fraction_from_onnx(int8) == 1.0


def test_quantized_model_stays_close_to_float32(exported, tmp_path) -> None:
    model, fp32 = exported
    int8 = tmp_path / "int8_pred.onnx"
    quantize_onnx_static(
        fp32,
        int8,
        calibration_reader=build_calibration_reader(_fake_loader(), limit=4),
    )
    x = torch.from_numpy(
        np.random.default_rng(2).standard_normal((16, CONTEXT_LEN, T, N_CHANNELS), dtype=np.float32)
    )

    with torch.no_grad():
        reference = model(x).argmax(dim=-1)
    quantized = OnnxModule(int8)(x).argmax(dim=-1)

    # An untrained model has near-tied logits, so this is a sanity bound rather
    # than an accuracy claim; real accuracy is measured on the test split.
    assert (reference == quantized).float().mean() > 0.5


def test_calibration_reader_stops_at_the_limit() -> None:
    reader = build_calibration_reader(_fake_loader(n_batches=10), limit=3)

    seen = 0
    while reader.get_next() is not None:
        seen += 1
    assert seen == 3


def test_int8_fraction_is_zero_for_a_graph_without_mac_operators(tmp_path) -> None:
    import onnx
    from onnx import helper, TensorProto

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "relu_only",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "relu.onnx"
    onnx.save(helper.make_model(graph), str(path))

    assert int8_mac_fraction_from_onnx(path) == 0.0
