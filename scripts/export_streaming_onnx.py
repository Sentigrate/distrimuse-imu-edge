"""Export split (encoder, temporal+head) ONNX graphs for streaming inference.

``inference/streaming.py``'s ``StreamingWindowPredictor`` implements
embedding-cached inference, but only for the PyTorch model in-process. A
partner consuming the ONNX-only ``models/imu`` package in
``distrimuse-ds-shared`` has no PyTorch dependency and no access to
``window_encoder``/``temporal`` as separate callables — the published ONNX
graphs are single end-to-end graphs (raw windows in, logits out).

This script exports two ONNX graphs per context mode and precision instead of
one combined graph:

- ``<label>_encoder_fp32.onnx``: ``window_encoder`` alone, one raw window
  ``(1, C, T)`` in, one embedding ``(1, D)`` out.
- ``<label>_temporal_<precision>.onnx``: ``temporal`` + position-select + ``head``,
  the full embedding buffer ``(1, D, total_context_len)`` in, logits
  ``(1, n_classes)`` out.

A caller can then reproduce ``StreamingWindowPredictor``'s ring-buffer
caching using two ``onnxruntime.InferenceSession``s instead of one PyTorch
module — see ``distrimuse-ds-shared/models/imu/pipeline.py``'s
``StreamingOnnxModel``, which is exactly that.

For static int8, the encoder and temporal graphs are calibrated separately on
the training split.  The generated pair is a real int8 streaming deployment
path, not a same-shapes memory estimate: the encoder's output is dequantized to
float at the ONNX boundary, then consumed by the int8 temporal graph.

Run from the repository root::

    uv run python scripts/export_streaming_onnx.py

Writes twelve ONNX files (3 context modes x 2 precisions x 2 graph parts) under
``experiments/exports/streaming_onnx/``. Float32 pairs are checked for exact
agreement with the combined checkpoint (up to floating-point reassociation).
The int8 pairs receive a separate ONNX Runtime smoke check here; downstream
equivalence is checked by the shared-package test against the published split
graphs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from distrimuse_imu_edge.compression.onnx_int8 import (
    DEFAULT_OPSET,
    quantize_onnx_static,
)
from distrimuse_imu_edge.data.config import data_config_from_mapping
from distrimuse_imu_edge.data.sequence import SequenceWindowDataset
from distrimuse_imu_edge.data.windowing import ChannelNormalizer
from distrimuse_imu_edge.training.runner import load_checkpoint_model, set_seed
from torch.utils.data import DataLoader

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
CALIBRATION_BATCHES = 32


class _NamedCalibrationReader:
    """ONNX Runtime calibration reader with a graph-specific input name."""

    def __init__(self, *, input_name: str, batches: Iterator[np.ndarray]) -> None:
        self._input_name = input_name
        self._batches = batches

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._batches, None)
        if batch is None:
            return None
        return {self._input_name: batch}


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


@dataclass
class _SplitCalibrationData:
    """The checkpoint's normalized training windows as sequence batches.

    The run may no longer have its source parquet split available, but its raw
    training-window cache is a stable, leakage-free calibration source.  Using
    that cache also recreates the normalizer fitted by ``IMUEdgeDataModule``.
    """

    dataset: SequenceWindowDataset
    batch_size: int

    def loader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)


def _encoder_calibration_batches(
    data: _SplitCalibrationData, *, limit: int
) -> Iterator[np.ndarray]:
    """Yield normalized individual windows as ``(B*N, C, T)`` for the CNN."""
    for index, (context, _, _) in enumerate(data.loader()):
        if index >= limit:
            return
        batch, windows, samples, channels = context.shape
        flat = context.reshape(batch * windows, samples, channels).transpose(1, 2)
        yield np.ascontiguousarray(flat.numpy(), dtype=np.float32)


def _temporal_calibration_batches(
    data: _SplitCalibrationData, model: nn.Module, *, limit: int
) -> Iterator[np.ndarray]:
    """Yield full float32 embedding buffers as ``(B, D, N)`` for the TCN."""
    model.eval()
    with torch.no_grad():
        for index, (context, _, _) in enumerate(data.loader()):
            if index >= limit:
                return
            embeddings = model.encode_windows(context).transpose(1, 2)
            yield np.ascontiguousarray(embeddings.numpy(), dtype=np.float32)


def _build_calibration_data(checkpoint: dict[str, Any]) -> _SplitCalibrationData:
    """Load the checkpoint's cached training windows for leakage-free PTQ."""
    data_cfg = data_config_from_mapping({"data": checkpoint["config"]["data"]})
    samples = int(round(data_cfg.window_size_s * 104))
    candidates: list[Path] = []
    for path in data_cfg.window_cache_dir.glob("train_*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            if "X" in payload and payload["X"].shape[1:] == (samples, len(data_cfg.sensor_cols)):
                candidates.append(path)
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in candidates) or "none"
        raise FileNotFoundError(
            "Expected exactly one compatible cached training-window archive for split ONNX "
            f"calibration, found: {found}."
        )
    with np.load(candidates[0], allow_pickle=False) as payload:
        x = payload["X"].astype(np.float32)
        y = payload["y"].astype(np.int64)
        person_ids = payload["person_ids"].astype(np.int64)
        scenario_ids = payload["scenario_ids"].astype(np.int64)
        starts = payload["window_starts_s"].astype(np.float64)
    normalized = ChannelNormalizer().fit(x).transform(x)
    return _SplitCalibrationData(
        dataset=SequenceWindowDataset(
            normalized,
            y,
            person_ids,
            scenario_ids,
            context_len=data_cfg.context_len,
            future_context_len=data_cfg.future_context_len,
            window_starts_s=starts,
        ),
        batch_size=data_cfg.batch_size,
    )


def _export_int8_pair(spec: dict, data: _SplitCalibrationData) -> tuple[Path, Path]:
    """Statically quantize independently exported CNN and temporal ONNX graphs."""
    label = str(spec["label"])
    out_dir = Path(spec["encoder_path"]).parent
    encoder_int8_path = out_dir / f"{label}_encoder_int8.onnx"
    temporal_int8_path = out_dir / f"{label}_temporal_int8.onnx"

    # Recreate the seeded training-loader order for each graph.  Both readers
    # see only the training split, while each observes the representation its
    # graph actually receives at runtime.
    set_seed(42)
    quantize_onnx_static(
        spec["encoder_path"],
        encoder_int8_path,
        calibration_reader=_NamedCalibrationReader(
            input_name=ENCODER_INPUT_NAME,
            batches=_encoder_calibration_batches(data, limit=CALIBRATION_BATCHES),
        ),
    )
    set_seed(42)
    quantize_onnx_static(
        spec["temporal_path"],
        temporal_int8_path,
        calibration_reader=_NamedCalibrationReader(
            input_name=TEMPORAL_INPUT_NAME,
            batches=_temporal_calibration_batches(data, spec["model"], limit=CALIBRATION_BATCHES),
        ),
    )
    return encoder_int8_path, temporal_int8_path


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
        "checkpoint": ckpt,
        "encoder_path": encoder_path,
        "temporal_path": temporal_path,
        "total_context_len": total_context_len,
        "current_index": int(model.current_index),
        "n_channels": n_channels,
        "t": t,
        "embedding_dim": embedding_dim,
    }


def verify_one(
    spec: dict,
    *,
    encoder_path: Path | None = None,
    temporal_path: Path | None = None,
    n_windows: int = 30,
    seed: int = 0,
) -> None:
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
        str(encoder_path or spec["encoder_path"]), options, providers=["CPUExecutionProvider"]
    )
    temporal_session = ort.InferenceSession(
        str(temporal_path or spec["temporal_path"]), options, providers=["CPUExecutionProvider"]
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


def smoke_int8_pair(spec: dict, *, encoder_path: Path, temporal_path: Path) -> None:
    """Check that a statically quantized split pair accepts a live window stream."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    encoder_session = ort.InferenceSession(
        str(encoder_path), options, providers=["CPUExecutionProvider"]
    )
    temporal_session = ort.InferenceSession(
        str(temporal_path), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    buffer: list[np.ndarray] = []
    for _ in range(int(spec["total_context_len"])):
        raw = rng.standard_normal((spec["t"], spec["n_channels"]), dtype=np.float32)
        x = np.ascontiguousarray(raw.T[None], dtype=np.float32)
        embedding = encoder_session.run(None, {ENCODER_INPUT_NAME: x})[0][0]
        buffer.append(np.asarray(embedding, dtype=np.float32))
    stacked = np.ascontiguousarray(np.stack(buffer, axis=0).T[None], dtype=np.float32)
    logits = temporal_session.run(None, {TEMPORAL_INPUT_NAME: stacked})[0][0]
    assert logits.shape == (9,), f"[{spec['label']}] unexpected int8 logits shape {logits.shape}"
    assert np.isfinite(logits).all(), f"[{spec['label']}] int8 pair emitted non-finite logits"
    print(f"[{spec['label']}] int8 split ONNX smoke check passed")


def main() -> None:
    for label, run_dir in RUNS:
        print(f"[{label}] exporting split ONNX graphs...")
        spec = export_one(label, run_dir, OUT_DIR)
        for key in ("encoder_path", "temporal_path"):
            path = spec[key]
            print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
        verify_one(spec)
        print(f"[{label}] calibrating split int8 graphs on the training split...")
        calibration_data = _build_calibration_data(spec["checkpoint"])
        encoder_int8_path, temporal_int8_path = _export_int8_pair(spec, calibration_data)
        for path in (encoder_int8_path, temporal_int8_path):
            print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
        # Separate static PTQ of the two graphs changes rounding points, so
        # full-graph and split-graph int8 logits need not agree numerically.
        smoke_int8_pair(spec, encoder_path=encoder_int8_path, temporal_path=temporal_int8_path)


if __name__ == "__main__":
    main()
