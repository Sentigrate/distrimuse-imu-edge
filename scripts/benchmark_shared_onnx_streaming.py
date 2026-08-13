"""Benchmark every published shared IMU ONNX variant on the configured test split.

This is the source of truth for the deployment mini-paper.  It evaluates the
combined and embedding-cached ONNX paths on the same held-out window cache and
uses the normalization parameters shipped in ``distrimuse-ds-shared``.  It also
times the actual ONNX Runtime cached path and inspects the published ONNX tensor
types/shapes to report graph-native activation peaks, rather than assuming that
static quantization simply divides every float32 activation by four.

Run from this repository::

    uv run python scripts/benchmark_shared_onnx_streaming.py

The output JSON is intentionally checked into the report assets so figures and
text can use one reproducible measurement record.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from sklearn.metrics import f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ROOT = REPO_ROOT.parent / "distrimuse-ds-shared"
DEFAULT_TEST_WINDOWS = REPO_ROOT / "cache/windows/test_e08d1a9b655d553c.npz"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/edge_window_tcn_context_report_assets/shared_onnx_streaming_benchmark.json"
)
WARMUP_CALLS = 100
TIMED_CALLS = 500
TIMING_TRIALS = 9
FULL_EVAL_BATCH = 64


def _add_shared_repo(shared_root: Path) -> None:
    root = str(shared_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _time_ms(
    fn: Callable[[], Any],
    *,
    warmup: int = WARMUP_CALLS,
    repeats: int = TIMED_CALLS,
    trials: int = TIMING_TRIALS,
) -> dict[str, float]:
    """Return a stable batch-one latency after session warm-up.

    Small static-int8 ONNX graphs can incur one-off kernel preparation costs
    beyond a short warm-up.  Reporting the median of several timed-trial
    medians keeps the deployment comparison from reflecting that transient.
    """
    for _ in range(warmup):
        fn()
    values: list[float] = []
    trial_medians: list[float] = []
    for _ in range(trials):
        trial_values: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            trial_values.append((time.perf_counter() - start) * 1000.0)
        trial_medians.append(float(statistics.median(trial_values)))
        values.extend(trial_values)
    values.sort()
    return {
        "median_ms": float(statistics.median(trial_medians)),
        "p95_ms": float(values[max(0, int(0.95 * len(values)) - 1)]),
        "warmup_calls": warmup,
        "timed_calls_per_trial": repeats,
        "timing_trials": trials,
    }


def _f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def _group_ids(payload: Any) -> np.ndarray:
    return np.asarray(
        [f"{person}:{scenario}" for person, scenario in zip(payload["person_ids"], payload["scenario_ids"])],
        dtype=object,
    )


def _validate_config_split(test_path: Path, payload: Any) -> None:
    split = yaml.safe_load((REPO_ROOT / "configs/split.yaml").read_text(encoding="utf-8"))["split"]
    expected_people = set(int(value) for value in split["test"])
    actual_people = set(int(value) for value in np.unique(payload["person_ids"]))
    if actual_people != expected_people:
        raise ValueError(
            f"{test_path} does not match configs/split.yaml test people: "
            f"expected {sorted(expected_people)}, found {sorted(actual_people)}."
        )


def _contexts_for_variant(windows: np.ndarray, group_ids: np.ndarray, variant: Any, build_context_sequences: Callable[..., np.ndarray]) -> np.ndarray:
    return build_context_sequences(
        windows,
        variant.context_len,
        current_index=variant.current_index,
        group_ids=group_ids,
    )


def _normal_predictions(model: Any, contexts: np.ndarray) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(contexts), FULL_EVAL_BATCH):
        batches.append(np.argmax(model.logits(contexts[start : start + FULL_EVAL_BATCH]), axis=1))
    return np.concatenate(batches)


def _cached_predictions(model: Any, windows: np.ndarray, labels: np.ndarray, groups: np.ndarray, variant: Any) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate only real contexts: no imaginary history or end-of-stream future."""
    delay = variant.context_len - 1 - variant.current_index
    predictions: list[int] = []
    targets: list[int] = []
    previous_group: object | None = None
    for newest_index, (raw, group) in enumerate(zip(windows, groups)):
        if group != previous_group:
            model.reset()
            previous_group = group
        logits = model.push(raw)
        if logits is None:
            continue
        predictions.append(int(np.argmax(logits)))
        targets.append(int(labels[newest_index - delay]))
    return np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)


def _normal_predictions_for_cached_targets(
    model: Any,
    windows: np.ndarray,
    groups: np.ndarray,
    variant: Any,
    build_context_for_target: Callable[..., np.ndarray],
) -> np.ndarray:
    """Reference combined-graph predictions for precisely the cached-valid windows."""
    delay = variant.context_len - 1 - variant.current_index
    target_indices: list[int] = []
    run_length = 0
    previous_group: object | None = None
    for newest_index, group in enumerate(groups):
        if group != previous_group:
            run_length = 0
            previous_group = group
        run_length += 1
        if run_length >= variant.context_len:
            target_indices.append(newest_index - delay)
    contexts = np.stack(
        [
            build_context_for_target(
                windows,
                target_index,
                variant.context_len,
                current_index=variant.current_index,
                group_ids=groups,
            )
            for target_index in target_indices
        ]
    )
    return _normal_predictions(model, contexts)


def _activation_peak_bytes(path: Path) -> int:
    """Measure the graph's actual node I/O activation footprint with ORT profiling.

    ONNX Runtime records ``activation_size`` and ``output_size`` for every
    executed CPU node.  Their sum is the concrete input-plus-output activation
    footprint for that node, so the maximum gives the report's established
    ping-pong-buffer definition.  Do not reconstruct this from all input type
    shapes: that list also contains constant weight tensors for convolution
    nodes.  The runtime fields keep weights, scales, and process-wide allocator
    reservation out of this activation metric while retaining real uint8/float
    transitions in static-PTQ graphs.
    """
    graph = onnx.load(path).graph
    input_value = graph.input[0]
    input_name = input_value.name
    input_shape = tuple(
        int(dim.dim_value) if dim.dim_value > 0 else 1
        for dim in input_value.type.tensor_type.shape.dim
    )
    input_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_value.type.tensor_type.elem_type)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    session.run(None, {input_name: np.zeros(input_shape, dtype=input_dtype)})
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    finally:
        profile_path.unlink(missing_ok=True)

    peak = 0
    for event in events:
        args = event.get("args", {})
        if "activation_size" not in args or "output_size" not in args:
            continue
        peak = max(
            peak,
            int(args["activation_size"]) + int(args["output_size"]),
        )
    if peak <= 0:
        raise ValueError(f"ONNX Runtime profiling yielded no activation tensors for {path}")
    return peak


def _encoder_embedding_bytes(path: Path, *, context_len: int) -> int:
    """Resident float32 ring-buffer state in ``StreamingOnnxModel``."""
    output = onnx.load(path).graph.output[0]
    shape = [int(dim.dim_value) if dim.dim_value > 0 else 1 for dim in output.type.tensor_type.shape.dim]
    itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type)).itemsize
    return int(np.prod(shape)) * itemsize * context_len


def _kib(value: int) -> float:
    return round(value / 1024.0, 3)


def _variant_row(
    *,
    key: str,
    variant: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    build_context_sequences: Callable[..., np.ndarray],
    build_context_for_target: Callable[..., np.ndarray],
    onnx_model: Any,
    streaming_model: Any,
) -> dict[str, Any]:
    normal = onnx_model(variant)
    cached = streaming_model(variant)
    contexts = _contexts_for_variant(windows, groups, variant, build_context_sequences)
    full_pred = _normal_predictions(normal, contexts)
    cached_targets, cached_pred = _cached_predictions(cached, windows, labels, groups, variant)
    cached_reference = _normal_predictions_for_cached_targets(
        normal, windows, groups, variant, build_context_for_target
    )
    if len(cached_targets) == 0:
        raise ValueError(f"{key} produced no valid cached predictions")
    full_f1 = _f1(labels, full_pred)
    if variant.test_macro_f1 is not None and not np.isclose(
        full_f1, variant.test_macro_f1, atol=5e-4
    ):
        raise AssertionError(
            f"{key} full-test F1 {full_f1:.6f} does not reproduce the shared config "
            f"value {variant.test_macro_f1:.6f}."
        )

    # Time a real held-out input after cache warmup. Reset before and after so
    # the timing sequence cannot change the reported F1 state.
    latency_context = contexts[len(contexts) // 2 : len(contexts) // 2 + 1]
    normal_timing = _time_ms(lambda: normal.logits(latency_context))
    cached.reset()
    for raw in windows[: variant.context_len - 1]:
        cached.push(raw)
    cached_timing = _time_ms(lambda: cached.push(windows[variant.context_len - 1]))
    cached.reset()

    normal_peak = _activation_peak_bytes(variant.path)
    encoder_peak = _activation_peak_bytes(variant.encoder_path)
    temporal_peak = _activation_peak_bytes(variant.temporal_path)
    cache_state = _encoder_embedding_bytes(variant.encoder_path, context_len=variant.context_len)
    cached_peak = max(encoder_peak + cache_state, temporal_peak + cache_state)
    return {
        "variant": key,
        "precision": variant.precision,
        "context_len": variant.context_len,
        "current_index": variant.current_index,
        "future_context_len": variant.delay_windows,
        "test_full_zero_padded": {
            "n_predictions": int(len(labels)),
            "macro_f1": full_f1,
            "matches_shared_config": True,
        },
        "test_stream_valid": {
            "n_predictions": int(len(cached_targets)),
            "macro_f1_cached": _f1(cached_targets, cached_pred),
            "macro_f1_combined_reference": _f1(cached_targets, cached_reference),
            "class_agreement_with_combined": float(np.mean(cached_pred == cached_reference)),
        },
        "latency_host_onnxruntime": {"normal": normal_timing, "cached": cached_timing},
        "activation_peak_graph_native": {
            "definition": "largest ONNX Runtime-profiled node input+output activation footprint; weights/scales excluded",
            "normal_bytes": normal_peak,
            "normal_kib": _kib(normal_peak),
            "cached_encoder_bytes": encoder_peak,
            "cached_temporal_bytes": temporal_peak,
            "cached_embedding_ring_buffer_bytes": cache_state,
            "cached_bytes": cached_peak,
            "cached_kib": _kib(cached_peak),
        },
        "onnx_artifact": {
            "normal_kib": round(variant.path.stat().st_size / 1024.0, 3),
            "cached_split_kib": round(
                (variant.encoder_path.stat().st_size + variant.temporal_path.stat().st_size)
                / 1024.0,
                3,
            ),
        },
    }


def main() -> None:
    shared_root = DEFAULT_SHARED_ROOT
    test_path = DEFAULT_TEST_WINDOWS
    output = DEFAULT_OUTPUT
    _add_shared_repo(shared_root)
    from models.imu.io import load_yaml
    from models.imu.pipeline import (
        OnnxModel,
        StreamingOnnxModel,
        build_channel_normalizer,
        build_context_for_target,
        build_context_sequences,
        load_model_registry,
    )

    cfg = load_yaml(shared_root / "models/config/imu_shared_config.yaml")
    registry = load_model_registry(cfg, shared_root)
    payload = np.load(test_path, allow_pickle=False)
    _validate_config_split(test_path, payload)
    windows = build_channel_normalizer(cfg).transform(payload["X"])
    labels = np.asarray(payload["y"], dtype=np.int64)
    groups = _group_ids(payload)

    rows: list[dict[str, Any]] = []
    for key, variant in registry.items():
        print(f"[{key}] test-set F1, ONNX Runtime latency, and graph-native activation peak...")
        rows.append(
            _variant_row(
                key=key,
                variant=variant,
                windows=windows,
                labels=labels,
                groups=groups,
                build_context_sequences=build_context_sequences,
                build_context_for_target=build_context_for_target,
                onnx_model=OnnxModel,
                streaming_model=StreamingOnnxModel,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "test_windows": str(test_path.relative_to(REPO_ROOT)),
                "test_people": sorted(set(int(value) for value in payload["person_ids"])),
                "normalization": "models/config/imu_shared_config.yaml preprocessing.normalization",
                "latency_protocol": (
                    f"{WARMUP_CALLS} warm-ups + {TIMING_TRIALS} trials of "
                    f"{TIMED_CALLS} timed calls on one held-out input"
                ),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
