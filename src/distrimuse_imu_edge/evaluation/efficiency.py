from __future__ import annotations

import io
import statistics
import time
from typing import Any, Mapping

import torch
from torch import nn

from distrimuse_imu_edge.evaluation.energy import (
    DEFAULT_HOP_SIZE_S,
    EnergyProfile,
    estimate_energy,
)

# Module paths whose layers hold int8 weights and run int8 kernels. Covers the
# eager-mode dynamic/static quantised modules and their fused variants.
_QUANTIZED_MODULE_PREFIXES = (
    "torch.ao.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
    "torch.nn.quantized",
    "torch.nn.intrinsic.quantized",
)


def _is_quantized_module(module: nn.Module) -> bool:
    return type(module).__module__.startswith(_QUANTIZED_MODULE_PREFIXES)


def _int8_mac_fraction(summary_result: Any) -> float:
    """Share of traced MACs that sit in genuinely int8 layers.

    Walks the leaf layers torchinfo recorded and attributes each layer's MACs to
    either the int8 or the float32 path. Only leaves are counted because
    torchinfo also reports container modules holding the aggregated MACs of
    their children, and summing those would double-count.

    Returning a measured fraction rather than trusting the compression label
    matters because PyTorch dynamic quantisation converts ``Linear``/``GRU``
    only. A convolution-dominated model can be labelled as quantised while
    essentially none of its arithmetic actually runs in int8, and treating it as
    fully int8 would overstate its efficiency several-fold.

    Falls back to ``0.0`` (all float32) whenever the per-layer data is missing
    or sums to zero, so an unreadable trace understates efficiency rather than
    inventing it.
    """
    try:
        leaves = [
            layer
            for layer in summary_result.summary_list
            if getattr(layer, "is_leaf_layer", False)
        ]
    except AttributeError:
        return 0.0
    total = 0
    int8 = 0
    for layer in leaves:
        layer_macs = int(getattr(layer, "macs", 0) or 0)
        if layer_macs <= 0:
            continue
        total += layer_macs
        if _is_quantized_module(layer.module):
            int8 += layer_macs
    if total <= 0:
        return 0.0
    return int8 / total


def _shape_elements(shape: Any) -> int | None:
    """Count elements in a torchinfo ``input_size``/``output_size`` value.

    The value is usually a flat list of dimensions for one tensor. For a layer
    with several tensor inputs (or outputs) torchinfo nests a list of such
    shapes instead, so this recurses one level and sums the parts. Returns
    ``None`` for anything unreadable — an empty shape, or a negative dimension,
    which torchinfo emits for a few recursive/repeated-layer cases — so a
    layer that cannot be sized is skipped rather than silently mis-sized.
    """
    if not shape:
        return None
    if isinstance(shape[0], (list, tuple)):
        total = 0
        for sub in shape:
            sub_elems = _shape_elements(sub)
            if sub_elems is None:
                return None
            total += sub_elems
        return total
    if any(not isinstance(d, int) or d < 0 for d in shape):
        return None
    total = 1
    for d in shape:
        total *= d
    return total


def _peak_activation_bytes(summary_result: Any) -> int | None:
    """Peak "ping-pong buffer" size across the traced forward pass, in bytes.

    Every layer needs its input and output activation tensors held in memory
    simultaneously — the input while the output is being computed. The peak
    activation memory is the largest such input-plus-output size across every
    layer in the network, following the definition used throughout embedded-ML
    literature (e.g. Saha et al., "Machine Learning for Microcontroller-Class
    Hardware") and matching the reference thesis this project's deployment
    hardware is documented against (see ``DEPLOYMENT_HARDWARE.md``).

    Two things this figure does **not** capture, both of which can only push
    the true peak higher, never lower:

    - **Residual branches.** A block computing ``main(x) + shortcut(x)`` needs
      ``shortcut(x)`` held in memory for the full duration of ``main(x)``'s
      layers, not just for one layer's input-output pair. The elementwise add
      itself is a tensor operation, not an ``nn.Module``, so it has no entry in
      the trace this walks. A real memory planner (TVM's USMP, TFLite Micro's
      arena allocator) performs full liveness analysis across the graph; this
      function does the same single-layer maximum the reference thesis reports
      (Table 4.1), not that.
    - **Batching from the encode-windows reshape.** Context models
      (`edge_window_tcn`, `edge_window_gru`, `causal_context_transformer_cnn`)
      flatten `(B, N, T, C)` into an `(B*N, T, C)` batch to run one shared
      encoder over every window at once. Tracing that shape — which this
      function does, for consistency with the `macs`/`gflops` fields computed
      from the same forward pass — sizes the encoder's ping-pong buffers for
      *all* `N` windows going through at once. A streaming firmware
      implementation that encodes one new window per hop and reuses cached
      embeddings for the rest (see `DEPLOYMENT_HARDWARE.md`) would see a
      peak roughly `N` times smaller for the encoder layers.

    Returns ``None`` if ``summary_result`` carries no readable per-layer shape
    data, so an untraceable model reports an absent figure rather than a wrong
    one.
    """
    try:
        leaves = [
            layer
            for layer in summary_result.summary_list
            if getattr(layer, "is_leaf_layer", False)
        ]
    except AttributeError:
        return None
    peak = 0
    for layer in leaves:
        in_elems = _shape_elements(layer.input_size)
        out_elems = _shape_elements(layer.output_size)
        if in_elems is None or out_elems is None:
            continue
        size = (in_elems + out_elems) * 4  # traced module is always float32
        if size > peak:
            peak = size
    return peak or None


def _serialized_size_mb(model: nn.Module) -> float:
    """Return the size of the model's weights in megabytes.

    Serialises the state dict (all weight tensors) into an in-memory buffer
    and reads back how many bytes were written. This matches the file size you
    would get from ``torch.save(model.state_dict(), path)`` and reflects the
    storage cost of shipping the model to a device.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / 1e6, 4)


def _cpu_latency_ms(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 30,
) -> dict[str, float]:
    """Measure wall-clock inference latency on CPU for a single forward pass.

    The model is moved to CPU and set to eval mode before timing. A short
    warmup phase runs first so that any one-time JIT compilation or memory
    allocation costs are excluded from the measurements.

    Args:
        model: The model to time. Will be moved to CPU.
        sample_input: A representative input tensor (batch size 1 is typical).
        warmup: Number of forward passes to discard before timing starts.
        repeats: Number of timed forward passes. Median and p95 are reported.

    Returns:
        Dict with ``cpu_latency_median_ms`` and ``cpu_latency_p95_ms``.
        Median is the primary metric — it is robust to occasional OS scheduling
        jitter. p95 gives a worst-case bound useful for real-time guarantees.
    """
    model_cpu = model.to("cpu").eval()
    sample = sample_input.to("cpu")
    with torch.no_grad():
        for _ in range(warmup):
            model_cpu(sample)
        values: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model_cpu(sample)
            values.append((time.perf_counter() - start) * 1000.0)
    return {
        "cpu_latency_median_ms": float(statistics.median(values)),
        "cpu_latency_p95_ms": float(sorted(values)[max(0, int(0.95 * len(values)) - 1)]),
    }


def compute_model_stats(
    model: nn.Module,
    *,
    context_len: int,
    future_context_len: int = 0,
    window_size_s: float,
    n_channels: int,
    fs: int = 104,
    compression: dict[str, Any] | None = None,
    latency_repeats: int = 30,
    hop_size_s: float = DEFAULT_HOP_SIZE_S,
    energy_profile: EnergyProfile | str | Mapping[str, Any] | None = None,
    int8_mac_fraction: float | None = None,
) -> dict[str, Any]:
    """Compute a full efficiency profile for a trained model.

    Constructs a dummy input with the same shape as real inference — one batch,
    ``context_len + future_context_len`` windows, each containing
    ``window_size_s * fs`` time steps across ``n_channels`` sensor channels —
    then runs three measurements:

    **Parameter count**
        Counted directly from ``model.parameters()``. Trainable parameters
        determine how much gradient memory training needs; total parameters
        determine inference memory.

    **Model size (MB)**
        The serialised weight size, i.e. how much storage the model needs
        on a device. See ``_serialized_size_mb``.

    **MACs / GFLOPs**
        Computed via ``torchinfo``, which traces the full forward pass and
        counts multiply-accumulate operations (MACs) for every layer type,
        including attention matmuls that a hand-written hook approach would
        miss. ``gflops`` follows the ML community convention of reporting
        GMACs (``macs / 1e9``) under the GFLOPs label, matching the
        early-fusion project. The ``flops`` field stores the strict value
        (``2 * macs``) if needed.

    **Peak activation memory (KiB)**
        The largest per-layer "input + output" activation size across the
        traced forward pass — the ping-pong buffer a device must reserve
        regardless of how small the weights are. See
        ``_peak_activation_bytes`` for the definition and its two documented
        blind spots (residual branches, and the encode-windows batching that
        context models use). Reported for the traced float32 module
        (``peak_activation_kib_fp32``) alongside a naive same-shapes-in-int8
        projection (``peak_activation_kib_int8_est``) that assumes every
        activation tensor is stored 4x smaller and ignores that some layer
        types (e.g. ``LayerNorm``) have no int8 kernel and would not actually
        shrink under quantization.

    **CPU latency**
        Wall-clock timing of a single forward pass on CPU. See
        ``_cpu_latency_ms``. Note that this is timed on whatever machine ran
        the job — a laptop or a training GPU host — so it is a relative
        comparison between models, not a figure for any target device.

    **Energy (analytic)**
        Millijoules per inference, average milliwatts, and battery life under
        a declared hardware profile. Derived from ``macs``, so it re-expresses
        the MAC count in interpretable units rather than adding a new ranking
        axis, and it excludes sensor and radio power. See
        ``distrimuse_imu_edge.evaluation.energy`` for the model and its limits.

    Args:
        model: Trained model. Will be moved to CPU and set to eval mode.
        context_len: Number of past-plus-current windows fed per prediction.
        future_context_len: Number of future look-ahead windows.
        window_size_s: Duration of each window in seconds.
        n_channels: Number of input sensor channels (e.g. 6 for IMU).
        fs: Sampling frequency in Hz (default 104 Hz for this dataset).
        compression: Optional dict describing any compression applied
            (e.g. ``{"method": "onnx_static_int8"}``). Stored as-is in the output.
            It is deliberately *not* used to choose int8 versus float32
            throughput — the int8 MAC share is measured from the traced layers
            instead, because a compression label says nothing about how much of
            the arithmetic the quantiser actually converted.
        latency_repeats: Number of timed passes for latency measurement.
        hop_size_s: Seconds between consecutive predictions at deployment, used
            as the energy model's duty-cycle period. Should match the dataset's
            hop size, since one prediction is emitted per hop.
        energy_profile: Hardware profile name, profile object, or override
            mapping for the energy estimate. ``None`` uses the default profile.
        int8_mac_fraction: Override for the measured int8 MAC share. Leave as
            ``None`` to measure it from ``model``'s traced layers, which is
            correct whenever ``model`` is the thing that ships. Pass a value
            when the deployed artifact is *not* the traced module — for example
            an int8 ONNX graph exported from a float32 module, where tracing the
            source would report 0.0 and understate the deployed efficiency.

    Returns:
        Dict suitable for writing to ``model_stats.json``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    total_context_len = context_len + future_context_len
    t = int(round(window_size_s * fs))
    sample = torch.zeros(1, total_context_len, t, n_channels, dtype=torch.float32)
    model_cpu = model.to("cpu").eval()

    macs: int | None = None
    measured_int8_fraction = 0.0
    peak_activation_bytes: int | None = None
    try:
        from torchinfo import summary as ti_summary

        result = ti_summary(model_cpu, input_data=sample, verbose=0)
        macs = int(result.total_mult_adds)
        measured_int8_fraction = _int8_mac_fraction(result)
        peak_activation_bytes = _peak_activation_bytes(result)
    except Exception:
        pass
    effective_int8_fraction = (
        measured_int8_fraction if int8_mac_fraction is None else float(int8_mac_fraction)
    )
    peak_activation_kib_fp32 = (
        None if peak_activation_bytes is None else round(peak_activation_bytes / 1024, 3)
    )
    peak_activation_kib_int8_est = (
        None if peak_activation_kib_fp32 is None else round(peak_activation_kib_fp32 / 4, 3)
    )

    stats: dict[str, Any] = {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "model_size_mb": _serialized_size_mb(model_cpu),
        "macs": macs,
        "flops": None if macs is None else int(2 * macs),
        "gmacs": None if macs is None else round(macs / 1e9, 6),
        "gflops": None if macs is None else round(macs / 1e9, 6),
        "peak_activation_bytes_fp32": peak_activation_bytes,
        "peak_activation_kib_fp32": peak_activation_kib_fp32,
        "peak_activation_kib_int8_est": peak_activation_kib_int8_est,
        "context_len": int(context_len),
        "future_context_len": int(future_context_len),
        "total_context_len": int(total_context_len),
        "input_shape": [1, int(total_context_len), t, int(n_channels)],
        "compression": compression or {"method": "none"},
        "energy": estimate_energy(
            macs=macs,
            hop_size_s=hop_size_s,
            profile=energy_profile,
            int8_mac_fraction=effective_int8_fraction,
        ),
    }
    stats.update(_cpu_latency_ms(model_cpu, sample, repeats=latency_repeats))
    return stats
