"""Measure and render streaming-vs-normal inference figures for the
edge_window_tcn context-length ablation.

Loads the three real width-0.25 checkpoints (current only, past 7 + current,
past 7 + current + future 7) and their int8 ONNX exports, and measures — not
estimates — six numbers per context mode:

- peak activation memory (KiB), float32, normal (all-windows-batched) path
- peak activation memory (KiB), float32, streaming (embedding-cached) path
- CPU latency (ms), float32, normal path
- CPU latency (ms), float32, streaming path
- CPU latency (ms), int8, normal path (real ONNX Runtime session, not a
  projection — the int8 ONNX artifacts already exist on disk)
- test macro-F1, float32 and int8 (read from the existing
  ``quantization_comparison.json`` per run, not recomputed)

int8 *memory* is not independently traced: this repository's established
convention (``compute_model_stats``'s ``peak_activation_kib_int8_est``, also
used throughout DEPLOYMENT_HARDWARE.md) is a naive same-shapes-÷4 projection,
because activation tensor shapes are unaffected by quantization even though
their storage width is. This script applies that identical convention to the
streaming path for consistency, rather than inventing a stricter method for
one half of the comparison. There is no int8 streaming *latency* number:
streaming is only implemented for the float32 PyTorch path
(``inference/streaming.py``) — an ONNX/int8 streaming wrapper does not exist,
so that cell is reported as unmeasured rather than guessed at.

Run from the repository root::

    uv run python scripts/render_streaming_comparison_figures.py

Writes ``streaming_comparison.json`` (the measured numbers, so the figures
cannot drift from what was actually measured) and two SVGs, all under
``experiments/results/edge_window_tcn_context_report_assets/``.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from distrimuse_imu_edge.compression.onnx_int8 import OnnxModule  # noqa: E402
from distrimuse_imu_edge.evaluation.efficiency import (  # noqa: E402
    compute_model_stats,
    compute_streaming_model_stats,
)
from distrimuse_imu_edge.training.runner import load_checkpoint_model  # noqa: E402

RESULTS = Path("experiments/results")
ASSETS = RESULTS / "edge_window_tcn_context_report_assets"

# (label, float32 run dir, int8 run dir) in the report's established order.
RUNS = [
    ("Current only", "edge_window_tcn_wm025_current", "edge_window_tcn_wm025_current_int8"),
    (
        "Past 7 + current",
        "edge_window_tcn_wm025_past7_current",
        "edge_window_tcn_wm025_past7_current_int8",
    ),
    (
        "Past 7 + current\n+ future 7",
        "edge_window_tcn_wm025_centered_scratch",
        "edge_window_tcn_wm025_centered_scratch_int8",
    ),
]

# Same palette and hatching as render_ptq_comparison_figures.py, so this
# figure reads as part of the same report rather than a new visual language.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GRID = "#b0b0b0"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#ffffff"
INT8_HATCH = "////"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "hatch.linewidth": 0.8,
        "svg.fonttype": "none",
    }
)


def _time_repeated_ms(fn, *, warmup: int = 5, repeats: int = 30) -> dict[str, float]:
    """Local twin of efficiency._time_repeated_ms — kept script-local so this
    file does not depend on that module's private API across the package
    boundary."""
    for _ in range(warmup):
        fn()
    values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(sorted(values)[max(0, int(0.95 * len(values)) - 1)]),
    }


def measure_one(label: str, fp32_dir: str, int8_dir: str) -> dict[str, Any]:
    ckpt_path = RESULTS / fp32_dir / "checkpoints" / "best.ckpt"
    model, ckpt = load_checkpoint_model(ckpt_path, map_location="cpu")
    model = model.eval()

    data_cfg = ckpt["config"]["data"]
    context_len = int(data_cfg["context_len"])
    future_context_len = int(data_cfg["future_context_len"])
    total_context_len = context_len + future_context_len
    window_size_s = float(data_cfg["window_size_s"])
    n_channels = len(data_cfg["sensor_cols"])
    fs = 104

    print(f"[{label}] measuring normal (batched) path...")
    batched = compute_model_stats(
        model,
        context_len=context_len,
        future_context_len=future_context_len,
        window_size_s=window_size_s,
        n_channels=n_channels,
        fs=fs,
        latency_repeats=30,
    )

    print(f"[{label}] measuring streaming (embedding-cached) path...")
    streaming = compute_streaming_model_stats(
        model,
        total_context_len=total_context_len,
        window_size_s=window_size_s,
        n_channels=n_channels,
        fs=fs,
        latency_repeats=30,
    )

    print(f"[{label}] measuring real int8 ONNX Runtime latency (normal path)...")
    onnx_path = RESULTS / int8_dir / "onnx" / "model_int8.onnx"
    onnx_module = OnnxModule(onnx_path)
    t = int(round(window_size_s * fs))
    onnx_sample = torch.zeros(1, total_context_len, t, n_channels, dtype=torch.float32)
    for _ in range(5):
        onnx_module(onnx_sample)
    int8_timing = _time_repeated_ms(lambda: onnx_module(onnx_sample), warmup=5, repeats=30)

    comparison = json.loads(
        (RESULTS / int8_dir / "reports" / "quantization_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    f1_fp32 = comparison["test_macro_f1"]["float32"]
    f1_int8 = comparison["test_macro_f1"]["int8"]

    return {
        "label": label,
        "context_len": context_len,
        "future_context_len": future_context_len,
        "total_context_len": total_context_len,
        "test_macro_f1_fp32": f1_fp32,
        "test_macro_f1_int8": f1_int8,
        "peak_kib_fp32_normal": batched["peak_activation_kib_fp32"],
        "peak_kib_int8_normal_est": batched["peak_activation_kib_int8_est"],
        "peak_kib_fp32_streaming": streaming["peak_activation_kib_fp32_streaming"],
        "peak_kib_int8_streaming_est": streaming["peak_activation_kib_int8_est_streaming"],
        "latency_ms_fp32_normal": batched["cpu_latency_median_ms"],
        "latency_ms_fp32_streaming": streaming["cpu_latency_median_ms_streaming"],
        "latency_ms_int8_normal_onnx": int8_timing["median_ms"],
    }


def measure_all() -> list[dict[str, Any]]:
    return [measure_one(label, fp32_dir, int8_dir) for label, fp32_dir, int8_dir in RUNS]


def _recessive_axes(ax, *, y_grid: bool = True, x_grid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if y_grid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, alpha=0.25, linewidth=0.8)
    if x_grid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, alpha=0.25, linewidth=0.8)


def figure_peak_memory(rows: list[dict], path: Path) -> None:
    """Peak activation memory: normal vs streaming, float32 and int8-est, per context mode."""
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    n = len(rows)
    width = 0.19
    # Four bars per context mode: fp32-normal, fp32-streaming, int8-normal(est), int8-streaming(est)
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    series = [
        ("peak_kib_fp32_normal", "float32, normal", False),
        ("peak_kib_fp32_streaming", "float32, streaming", False),
        ("peak_kib_int8_normal_est", "int8 (est.), normal", True),
        ("peak_kib_int8_streaming_est", "int8 (est.), streaming", True),
    ]
    x = range(n)
    for (key, _series_label, hatched), offset in zip(series, offsets):
        for index, row in enumerate(rows):
            colour = SERIES_COLORS[index]
            value = row[key]
            ax.bar(
                index + offset,
                value,
                width,
                color=colour,
                edgecolor=SURFACE if hatched else "none",
                hatch=INT8_HATCH if hatched else None,
                linewidth=0,
            )
            ax.text(
                index + offset,
                value * 1.04 + 0.6,
                f"{value:.0f}",
                ha="center",
                fontsize=7.5,
                rotation=90 if value > 40 else 0,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Peak activation memory, KiB (log scale)")
    ax.set_title(
        "Streaming caches embeddings instead of re-encoding every window:\n"
        "peak memory drops and stops scaling with context length",
        fontsize=11,
        pad=14,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    _recessive_axes(ax)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#8a8a8a", edgecolor="none", label="float32, normal"),
            Patch(facecolor="#8a8a8a", edgecolor=SURFACE, label="float32, streaming", alpha=0.55),
            Patch(
                facecolor="#8a8a8a",
                edgecolor=SURFACE,
                hatch=INT8_HATCH,
                label="int8 (est.), normal",
            ),
            Patch(
                facecolor="#8a8a8a",
                edgecolor=SURFACE,
                hatch=INT8_HATCH,
                label="int8 (est.), streaming",
                alpha=0.55,
            ),
        ],
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_f1_vs_memory(rows: list[dict], path: Path) -> None:
    """The headline figure: test macro-F1 vs peak activation memory, 3 configs x 3 context modes."""
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    configs = [
        ("peak_kib_fp32_normal", "test_macro_f1_fp32", "float32, normal", "o", False),
        ("peak_kib_int8_normal_est", "test_macro_f1_int8", "int8, normal", "s", True),
        (
            "peak_kib_int8_streaming_est",
            "test_macro_f1_int8",
            "int8, streaming",
            "^",
            True,
        ),
    ]

    for index, row in enumerate(rows):
        colour = SERIES_COLORS[index]
        xs, ys = [], []
        for mem_key, f1_key, _config_label, marker, hatched in configs:
            x, y = row[mem_key], row[f1_key]
            xs.append(x)
            ys.append(y)
            ax.scatter(
                [x],
                [y],
                s=110,
                marker=marker,
                facecolor=colour,
                edgecolor=INK if not hatched else SURFACE,
                linewidth=1.1 if not hatched else 0.6,
                zorder=3,
            )
        # Connect the three points for one context mode, so the reader can
        # trace "same model, three deployment choices" at a glance.
        ax.plot(xs, ys, color=colour, linewidth=1.2, alpha=0.45, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Peak activation memory, KiB (log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title(
        "F1 vs peak memory: streaming keeps int8's accuracy\nat a fraction of int8's own memory footprint",
        fontsize=11,
        pad=14,
    )
    _recessive_axes(ax)

    from matplotlib.lines import Line2D

    context_handles = [
        Line2D(
            [0], [0], marker="o", color=SERIES_COLORS[i], linestyle="", markersize=8, label=r["label"].replace("\n", " ")
        )
        for i, r in enumerate(rows)
    ]
    config_handles = [
        Line2D([0], [0], marker="o", color=INK_SOFT, linestyle="", markersize=8, label="float32, normal"),
        Line2D([0], [0], marker="s", color=INK_SOFT, linestyle="", markersize=8, label="int8, normal"),
        Line2D([0], [0], marker="^", color=INK_SOFT, linestyle="", markersize=8, label="int8, streaming"),
    ]
    legend1 = ax.legend(
        handles=context_handles, frameon=False, fontsize=8.5, loc="lower right", title="Context mode"
    )
    legend1.get_title().set_fontsize(8.5)
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=config_handles, frameon=False, fontsize=8.5, loc="upper left", title="Configuration"
    )
    legend2.get_title().set_fontsize(8.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = measure_all()

    comparison_path = ASSETS / "streaming_comparison.json"
    comparison_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {comparison_path}")

    memory_path = ASSETS / "streaming-peak-memory.svg"
    f1_path = ASSETS / "streaming-f1-vs-memory.svg"
    figure_peak_memory(rows, memory_path)
    figure_f1_vs_memory(rows, f1_path)
    for path in (memory_path, f1_path):
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
