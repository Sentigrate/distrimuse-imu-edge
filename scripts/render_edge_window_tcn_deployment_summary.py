"""Render presentation-ready Edge Window TCN deployment-summary figures.

The script reads the unified held-out-test benchmark produced by
``benchmark_shared_onnx_streaming.py``. This keeps the mini-paper figures
traceable to actual ONNX Runtime paths, including the int8 cache.

Run from ``distrimuse-imu-edge`` with::

    uv run python scripts/render_edge_window_tcn_deployment_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402


RESULTS = Path("experiments/results")
ASSETS = RESULTS / "edge_window_tcn_context_report_assets"
BENCHMARK_DATA = ASSETS / "shared_onnx_streaming_benchmark.json"

CONTEXTS = (
    ("Current only", "current", "#2a78d6"),
    ("Past 7 + current", "past7", "#eb6834"),
    (
        "Past 7 + current + future 7",
        "past7_future7",
        "#1baf7a",
    ),
)

INK = "#15222d"
MUTED = "#5c6974"
GRID = "#cbd5dc"
SURFACE = "#ffffff"
HIGHLIGHT = "#f4b942"
PAST = "#3c9d9b"
CURRENT = "#f28e2b"
FUTURE = "#8e6bbe"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
    }
)


def load_rows() -> list[dict]:
    """Join measured float32/int8 normal and cached ONNX deployment results."""
    rows_by_variant = {
        row["variant"]: row
        for row in json.loads(BENCHMARK_DATA.read_text(encoding="utf-8"))["rows"]
    }
    rows: list[dict] = []
    for label, variant_prefix, colour in CONTEXTS:
        fp32 = rows_by_variant[f"{variant_prefix}_fp32"]
        int8 = rows_by_variant[f"{variant_prefix}_int8"]
        rows.append(
            {
                "label": label,
                "colour": colour,
                "f1_fp32": fp32["test_full_zero_padded"]["macro_f1"],
                "f1_int8": int8["test_full_zero_padded"]["macro_f1"],
                "f1_fp32_cached": fp32["test_stream_valid"]["macro_f1_cached"],
                "f1_int8_cached": int8["test_stream_valid"]["macro_f1_cached"],
                "latency_fp32_normal": fp32["latency_host_onnxruntime"]["normal"]["median_ms"],
                "latency_fp32_cached": fp32["latency_host_onnxruntime"]["cached"]["median_ms"],
                "latency_int8_normal": int8["latency_host_onnxruntime"]["normal"]["median_ms"],
                "latency_int8_cached": int8["latency_host_onnxruntime"]["cached"]["median_ms"],
                "memory_fp32_normal": fp32["activation_peak_graph_native"]["normal_kib"],
                "memory_fp32_cached": fp32["activation_peak_graph_native"]["cached_kib"],
                "memory_int8_normal": int8["activation_peak_graph_native"]["normal_kib"],
                "memory_int8_cached": int8["activation_peak_graph_native"]["cached_kib"],
                "size_fp32_kib": fp32["onnx_artifact"]["normal_kib"],
                "size_int8_kib": int8["onnx_artifact"]["normal_kib"],
                "size_fp32_cached_kib": fp32["onnx_artifact"]["cached_split_kib"],
                "size_int8_cached_kib": int8["onnx_artifact"]["cached_split_kib"],
            }
        )
    return rows


def soften_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, alpha=0.55, linewidth=0.8)


def dot(ax: plt.Axes, x: float, y: float, *, colour: str, marker: str, size: float = 70, **kwargs: object) -> None:
    ax.scatter(x, y, s=size, marker=marker, facecolor=colour, edgecolor=SURFACE, linewidth=1.3, zorder=4, **kwargs)


def add_context_labels(ax: plt.Axes, rows: list[dict], x_key: str, y_key: str, *, log_x: bool) -> None:
    """Use compact direct labels only on the three central float32 points."""
    offsets = [(7, 7), (7, -15), (7, 7)]
    for row, (dx, dy) in zip(rows, offsets):
        short = {"Current only": "1 window", "Past 7 + current": "7 past + current"}.get(
            row["label"], "7 past + current + 7 future"
        )
        ax.annotate(
            short,
            (row[x_key], row[y_key]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color=row["colour"],
            weight="bold" if row["label"].endswith("future 7") else "normal",
        )
    if log_x:
        ax.set_xscale("log")


def deployment_tradeoffs(rows: list[dict], path: Path) -> None:
    """Reference-style three-panel F1 vs deployment-cost overview."""
    # One wide panel per row is deliberately presentation-friendly: labels,
    # annotations, and the three deployment forms remain readable on a slide.
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 14.5))
    fig.suptitle(
        "Edge Window TCN (width 0.25): accuracy versus deployment cost",
        fontsize=16,
        weight="bold",
        x=0.055,
        y=0.992,
        ha="left",
    )
    fig.text(
        0.055,
        0.967,
        "Held-out configured test split × float32 / static int8. Triangles show measured embedding-cached ONNX Runtime execution.",
        color=MUTED,
        fontsize=9.3,
    )

    # A — latency. All four deployed paths are measured in ONNX Runtime.
    ax = axes[0]
    for row in rows:
        c = row["colour"]
        ax.plot([row["latency_fp32_normal"], row["latency_fp32_cached"]], [row["f1_fp32"], row["f1_fp32_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        ax.plot([row["latency_int8_normal"], row["latency_int8_cached"]], [row["f1_int8"], row["f1_int8_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        dot(ax, row["latency_fp32_normal"], row["f1_fp32"], colour=c, marker="o")
        dot(ax, row["latency_int8_normal"], row["f1_int8"], colour=c, marker="s")
        dot(ax, row["latency_fp32_cached"], row["f1_fp32_cached"], colour=c, marker="v")
        dot(ax, row["latency_int8_cached"], row["f1_int8_cached"], colour=c, marker="^")
    add_context_labels(ax, rows, "latency_fp32_normal", "f1_fp32", log_x=True)
    ax.set_title("A. CPU latency", loc="left", weight="bold", fontsize=11)
    ax.set_xlabel("Median inference latency (ms, log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.text(
        0.02,
        0.02,
        "All timings: 20 warm-ups + 100 calls\non one held-out test input.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
        va="bottom",
    )
    soften_axes(ax)

    # B — activation memory.  The highlighted point is the recommended deployment configuration.
    ax = axes[1]
    for row in rows:
        c = row["colour"]
        ax.plot([row["memory_fp32_normal"], row["memory_fp32_cached"]], [row["f1_fp32"], row["f1_fp32_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        ax.plot([row["memory_int8_normal"], row["memory_int8_cached"]], [row["f1_int8"], row["f1_int8_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        dot(ax, row["memory_fp32_normal"], row["f1_fp32"], colour=c, marker="o")
        dot(ax, row["memory_int8_normal"], row["f1_int8"], colour=c, marker="s")
        dot(ax, row["memory_fp32_cached"], row["f1_fp32_cached"], colour=c, marker="v")
        dot(ax, row["memory_int8_cached"], row["f1_int8_cached"], colour=c, marker="^")
    best = rows[-1]
    ax.scatter(
        best["memory_int8_cached"],
        best["f1_int8_cached"],
        s=190,
        marker="o",
        facecolor="none",
        edgecolor=HIGHLIGHT,
        linewidth=2.6,
        zorder=5,
    )
    ax.annotate(
        "best streamed F1\nat minimum memory",
        (best["memory_int8_cached"], best["f1_int8_cached"]),
        xytext=(31, -15),
        textcoords="offset points",
        fontsize=8.3,
        color=INK,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": HIGHLIGHT, "lw": 1.5},
    )
    add_context_labels(ax, rows, "memory_fp32_normal", "f1_fp32", log_x=True)
    ax.set_title("B. Peak activation memory", loc="left", weight="bold", fontsize=11)
    ax.set_xlabel("Peak activation memory (KiB, log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.text(
        0.02,
        0.02,
        "Actual ONNX Runtime-profiled graph activations\nplus the resident embedding ring buffer.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
        va="bottom",
    )
    soften_axes(ax)

    # C — artifact storage. Split graphs are stored separately for caching.
    ax = axes[2]
    for row in rows:
        c = row["colour"]
        ax.plot([row["size_fp32_kib"], row["size_fp32_cached_kib"]], [row["f1_fp32"], row["f1_fp32_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        ax.plot([row["size_int8_kib"], row["size_int8_cached_kib"]], [row["f1_int8"], row["f1_int8_cached"]], color=c, alpha=0.45, linewidth=1.3, zorder=2)
        dot(ax, row["size_fp32_kib"], row["f1_fp32"], colour=c, marker="o")
        dot(ax, row["size_int8_kib"], row["f1_int8"], colour=c, marker="s")
        dot(ax, row["size_fp32_cached_kib"], row["f1_fp32_cached"], colour=c, marker="v")
        dot(ax, row["size_int8_cached_kib"], row["f1_int8_cached"], colour=c, marker="^")
    add_context_labels(ax, rows, "size_fp32_kib", "f1_fp32", log_x=False)
    ax.set_title("C. Exported ONNX size", loc="left", weight="bold", fontsize=11)
    ax.set_xlabel("Exported ONNX artifact size (KiB)")
    ax.set_ylabel("Test macro-F1")
    ax.text(
        0.02,
        0.02,
        "Cached deployment stores a split encoder + temporal graph pair\n(the table in the mini-paper reports both exact sizes).",
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
        va="bottom",
    )
    soften_axes(ax)

    for ax in axes:
        ax.set_ylim(0.455, 0.72)
        ax.set_yticks([0.46, 0.52, 0.58, 0.64, 0.70])

    context_legend = [
        Line2D([0], [0], marker="o", color=row["colour"], linestyle="", markersize=7, label=row["label"])
        for row in rows
    ]
    form_legend = [
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, linestyle="", markersize=7, label="float32, normal"),
        Line2D([0], [0], marker="s", color=INK, markerfacecolor=INK, linestyle="", markersize=7, label="int8, normal"),
        Line2D(
            [0],
            [0],
            marker="^",
            color=INK,
            markerfacecolor=INK,
            linestyle="",
            markersize=7,
            label="int8, cached embeddings",
        ),
        Line2D(
            [0], [0], marker="v", color=INK, markerfacecolor=INK,
            linestyle="", markersize=7, label="float32, cached embeddings",
        ),
    ]
    fig.legend(
        handles=context_legend + form_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12, hspace=0.48)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def label_box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, *, face: str, edge: str, fontsize: float = 8.2) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)


def cache_explainer(rows: list[dict], path: Path) -> None:
    """Explain the stateful schedule and quantify its future-context memory gain."""
    future = rows[-1]
    fig = plt.figure(figsize=(14.8, 7.2), constrained_layout=True)
    mosaic = fig.subplot_mosaic([["schedule", "schedule", "memory"], ["schedule", "schedule", "memory"]], width_ratios=[1.0, 1.0, 0.92])
    ax = mosaic["schedule"]
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 12.0)
    ax.axis("off")
    ax.set_title("A. Embedding cache: one new window per hop", loc="left", weight="bold", fontsize=12, pad=8)

    ax.text(0.0, 11.2, "Full context for the prediction of window t", fontsize=9.4, weight="bold")
    ax.text(0.0, 10.7, "7 past + current + 7 future; 3 s windows, 1 s hop", fontsize=8.3, color=MUTED)
    x0, box_w, gap = 0.25, 0.88, 0.12
    labels = [f"t{n:+d}".replace("+0", "") for n in range(-7, 8)]
    colours = [PAST] * 7 + [CURRENT] + [FUTURE] * 7
    for index, (label, colour) in enumerate(zip(labels, colours)):
        x = x0 + index * (box_w + gap)
        label_box(ax, x, 9.3, box_w, 0.72, label, face=colour + "33", edge=colour, fontsize=7.6)
    ax.text(7.82, 10.25, "current\nclassified", ha="center", va="bottom", fontsize=7.3, color=CURRENT, weight="bold")
    ax.text(14.9, 10.15, "7 s look-ahead\nremains required", ha="center", va="bottom", fontsize=7.3, color=FUTURE, weight="bold")

    ax.text(0.0, 8.4, "Naive batched inference", fontsize=9.6, weight="bold", color=INK)
    ax.text(0.0, 7.92, "At every hop: re-encode all 15 raw windows", fontsize=8.2, color=MUTED)
    for index, colour in enumerate(colours):
        x = x0 + index * (box_w + gap)
        label_box(ax, x, 6.82, box_w, 0.72, "CNN", face=colour + "2e", edge=colour, fontsize=7.2)
    ax.add_patch(FancyArrowPatch((15.2, 7.18), (16.2, 7.18), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
    label_box(ax, 16.3, 6.72, 1.55, 0.91, "TCN\n→ t", face="#edf0ff", edge="#465cc7", fontsize=7.2)
    ax.text(0.0, 6.08, f"15 encoder passes per 1 s hop  ·  {future['memory_fp32_normal']:.1f} KiB measured peak", fontsize=8.5, color=INK, weight="bold")

    ax.text(0.0, 4.84, "Embedding-cached streaming", fontsize=9.6, weight="bold", color=INK)
    ax.text(0.0, 4.36, "At the next hop: encode only the arriving future window", fontsize=8.2, color=MUTED)
    label_box(ax, x0 + 14 * (box_w + gap), 3.22, box_w, 0.72, "CNN", face=FUTURE + "2e", edge=FUTURE, fontsize=7.2)
    ax.text(x0 + 14 * (box_w + gap) + box_w / 2, 2.96, "new", ha="center", va="top", fontsize=7.3, color=FUTURE, weight="bold")
    ax.add_patch(FancyArrowPatch((15.15, 3.58), (14.2, 3.58), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.1))
    for index, colour in enumerate(colours):
        x = x0 + index * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 1.92), box_w, 0.68, facecolor=colour + "55", edgecolor=colour, linewidth=0.9))
        ax.text(x + box_w / 2, 2.26, "e", ha="center", va="center", fontsize=7.1, color=INK)
    ax.text(7.75, 1.38, "rolling buffer of 15 cached 24-D embeddings", ha="center", fontsize=8.1, color=INK)
    ax.add_patch(FancyArrowPatch((15.2, 2.26), (16.2, 2.26), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
    label_box(ax, 16.3, 1.78, 1.55, 0.96, "TCN\n→ t", face="#edf0ff", edge="#465cc7", fontsize=7.2)
    ax.text(0.0, 0.62, f"1 encoder pass per hop  ·  {future['memory_fp32_cached']:.1f} KiB measured peak  ·  same valid-stream classes", fontsize=8.5, color=INK, weight="bold")

    ax = mosaic["memory"]
    ax.set_title("B. Why caching makes the future model fit", loc="left", weight="bold", fontsize=12, pad=8)
    categories = ["float32", "int8"]
    normal = [future["memory_fp32_normal"], future["memory_int8_normal"]]
    cached = [future["memory_fp32_cached"], future["memory_int8_cached"]]
    x = [0, 1]
    width = 0.31
    normal_bars = ax.bar([v - width / 2 for v in x], normal, width, color="#aab7c2", label="normal: re-encode all windows")
    cached_bars = ax.bar([v + width / 2 for v in x], cached, width, color=future["colour"], label="cached embeddings")
    ax.set_yscale("log")
    ax.set_xticks(x, categories)
    ax.set_ylabel("Peak activation memory (KiB, log scale)")
    ax.yaxis.grid(True, color=GRID, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    for bars, values in ((normal_bars, normal), (cached_bars, cached)):
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:g}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5, weight="bold")
    ax.text(
        0.5,
        0.79,
        "≈13× lower peak\nmemory for the\n15-window model",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.48,
        "Same class predictions on\nthe valid stream; boundary\nwindows are not zero-padded.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )
    ax.legend(frameon=False, fontsize=7.7, loc="lower left")
    fig.suptitle("Caching embeddings removes repeated CNN work — not the 7 s look-ahead", fontsize=16, weight="bold", x=0.04, ha="left")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    deployment_tradeoffs(rows, ASSETS / "edge-window-tcn-deployment-tradeoffs.svg")
    cache_explainer(rows, ASSETS / "edge-window-tcn-embedding-cache-explainer.svg")
    print("Wrote Edge Window TCN deployment summary figures.")


if __name__ == "__main__":
    main()
