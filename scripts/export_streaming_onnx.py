"""Export split (encoder, temporal+head) ONNX graphs for streaming inference.

``inference/streaming.py``'s ``StreamingWindowPredictor`` implements
embedding-cached inference, but only for the PyTorch model in-process. A
partner consuming the ONNX-only ``models/imu`` package in
``distrimuse-ds-shared`` has no PyTorch dependency and no access to
``window_encoder``/``temporal`` as separate callables — the published ONNX
graphs are single end-to-end graphs (raw windows in, logits out).

This script exports two ONNX graphs per context mode instead of one:

- ``<label>_encoder_fp32.onnx``: ``window_encoder`` alone, one raw window
  ``(1, C, T)`` in, one embedding ``(1, D)`` out.
- ``<label>_temporal_fp32.onnx``: ``temporal`` + position-select + ``head``,
  the full embedding buffer ``(1, D, total_context_len)`` in, logits
  ``(1, n_classes)`` out.

A caller can then reproduce ``StreamingWindowPredictor``'s ring-buffer
caching using two ``onnxruntime.InferenceSession``s instead of one PyTorch
module — see ``distrimuse-ds-shared/models/imu/pipeline.py``'s
``StreamingOnnxModel``, which is exactly that.

int8 is intentionally out of scope here: the source PyTorch implementation
(``inference/streaming.py``) is float32-only too (see
``experiments/results/edge_window_tcn_context_comparison.md``'s "Streaming,
embedding-cached inference" section), so this keeps parity rather than
introducing a streaming/int8 combination that has not been measured anywhere.

Run from the repository root::

    uv run python scripts/export_streaming_onnx.py

Writes six ONNX files (3 context modes x 2 graph parts) under
``experiments/exports/streaming_onnx/``, and verifies each pair reproduces
the combined checkpoint's batched ``forward()`` output exactly (up to
floating-point reassociation) when fed the same window stream one window at
a time — the same equivalence ``tests/test_streaming_equivalence.py`` checks
for the in-process PyTorch path, checked again here for the exported graphs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from distrimuse_imu_edge.compression.onnx_int8 import DEFAULT_OPSET
from distrimuse_imu_edge.training.runner import load_checkpoint_model

RESULTS = Path("experiments/results")
OUT_DIR = Path("experiments/exports/streaming_onnx")

# (label, checkpoint dir) — the three real width-0.25 context modes.
RUNS = [
    ("current", "edge_window_tcn_wm025_current"),
    ("past7", "edge_window_tcn_wm025_past7_current"),
    ("past7_future7", "edge_window_tcn_wm025_centered_scratch"),
]

ENCODER_INPUT_NAME = "window"
ENCODER_OUTPUT_NAME = "embedding"
TEMPORAL_INPUT_NAME = "embeddings"
TEMPORAL_OUTPUT_NAME = "logits"


class _TemporalHead(nn.Module):
    """``temporal`` + current-position select + ``head``, as one exportable module.

    Exactly ``EdgeWindowTCN.forward``'s second half — see
    ``models/edge_window_sequence.py`` — just starting from a precomputed
    embedding buffer instead of raw windows, so it can be exported and run
    independently of ``window_encoder``.
    """

    def __init__(self, temporal: nn.Module, head: nn.Module, current_index: int) -> None:
        super().__init__()
        self.temporal = temporal
        self.head = head
        self.current_index = current_index

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        temporal_out = self.temporal(embeddings).transpose(1, 2)  # (B, N, D)
        return self.head(temporal_out[:, self.current_index])


def export_one(label: str, run_dir: str, out_dir: Path) -> dict:
    ckpt_path = RESULTS / run_dir / "checkpoints" / "best.ckpt"
    model, ckpt = load_checkpoint_model(ckpt_path, map_location="cpu")
    model = model.eval()

    data_cfg = ckpt["config"]["data"]
    context_len = int(data_cfg["context_len"])
    future_context_len = int(data_cfg["future_context_len"])
    total_context_len = context_len + future_context_len
    window_size_s = float(data_cfg["window_size_s"])
    n_channels = len(data_cfg["sensor_cols"])
    fs = 104
    t = int(round(window_size_s * fs))

    with torch.no_grad():
        embedding_dim = int(model.window_encoder(torch.zeros(1, n_channels, t)).shape[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / f"{label}_encoder_fp32.onnx"
    temporal_path = out_dir / f"{label}_temporal_fp32.onnx"

    encoder_sample = torch.zeros(1, n_channels, t, dtype=torch.float32)
    torch.onnx.export(
        model.window_encoder,
        (encoder_sample,),
        str(encoder_path),
        input_names=[ENCODER_INPUT_NAME],
        output_names=[ENCODER_OUTPUT_NAME],
        dynamic_axes={
            ENCODER_INPUT_NAME: {0: "batch"},
            ENCODER_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=DEFAULT_OPSET,
        dynamo=False,
    )

    temporal_head = _TemporalHead(model.temporal, model.head, int(model.current_index)).eval()
    temporal_sample = torch.zeros(1, embedding_dim, total_context_len, dtype=torch.float32)
    torch.onnx.export(
        temporal_head,
        (temporal_sample,),
        str(temporal_path),
        input_names=[TEMPORAL_INPUT_NAME],
        output_names=[TEMPORAL_OUTPUT_NAME],
        dynamic_axes={
            TEMPORAL_INPUT_NAME: {0: "batch"},
            TEMPORAL_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=DEFAULT_OPSET,
        dynamo=False,
    )

    return {
        "label": label,
        "model": model,
        "encoder_path": encoder_path,
        "temporal_path": temporal_path,
        "total_context_len": total_context_len,
        "current_index": int(model.current_index),
        "n_channels": n_channels,
        "t": t,
        "embedding_dim": embedding_dim,
    }


def verify_one(spec: dict, *, n_windows: int = 30, seed: int = 0) -> None:
    """Feed the same random window stream through PyTorch (batched) and the
    exported ONNX pair (streamed, one window at a time); assert every emitted
    prediction matches, mirroring tests/test_streaming_equivalence.py."""
    from collections import deque

    torch.manual_seed(seed)
    total_context_len = spec["total_context_len"]
    current_index = spec["current_index"]
    n_channels = spec["n_channels"]
    t = spec["t"]
    model = spec["model"]

    options = ort.SessionOptions()
    options.log_severity_level = 3
    encoder_session = ort.InferenceSession(
        str(spec["encoder_path"]), options, providers=["CPUExecutionProvider"]
    )
    temporal_session = ort.InferenceSession(
        str(spec["temporal_path"]), options, providers=["CPUExecutionProvider"]
    )

    session = torch.randn(total_context_len + n_windows, t, n_channels)
    buffer: deque[np.ndarray] = deque(maxlen=total_context_len)
    delay = total_context_len - 1 - current_index
    checked = 0

    for i in range(session.shape[0]):
        raw = session[i]
        x = raw.transpose(0, 1).unsqueeze(0).numpy().astype(np.float32)  # (1, C, T)
        embedding = encoder_session.run(None, {ENCODER_INPUT_NAME: x})[0][0]  # (D,)
        buffer.append(embedding)
        if len(buffer) < total_context_len:
            continue

        stacked = np.stack(list(buffer), axis=0).T[None].astype(np.float32)  # (1, D, N)
        logits = temporal_session.run(None, {TEMPORAL_INPUT_NAME: stacked})[0][0]

        target_abs_index = i - delay
        start = target_abs_index - current_index
        end = start + total_context_len
        context = session[start:end].unsqueeze(0)
        with torch.no_grad():
            expected = model(context).squeeze(0).numpy()

        assert np.allclose(logits, expected, atol=1e-4), (
            f"[{spec['label']}] streaming ONNX and batched PyTorch diverge at step {i}: "
            f"{logits} vs {expected}"
        )
        checked += 1

    # Warmup consumes total_context_len-1 windows before the first check; every
    # remaining pushed window (n_windows of them, plus the one that completed
    # warmup) yields one checked prediction.
    expected_checks = n_windows + 1
    assert checked == expected_checks, (
        f"[{spec['label']}] expected {expected_checks} checks, got {checked}"
    )
    print(f"[{spec['label']}] verified {checked} streamed predictions match batched forward() exactly")


def main() -> None:
    for label, run_dir in RUNS:
        print(f"[{label}] exporting split ONNX graphs...")
        spec = export_one(label, run_dir, OUT_DIR)
        for key in ("encoder_path", "temporal_path"):
            path = spec[key]
            print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
        verify_one(spec)


if __name__ == "__main__":
    main()
