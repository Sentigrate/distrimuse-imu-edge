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


def transition_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str,
    linestyle: str,
) -> None:
    """Show a deployment transition while leaving both endpoints visible."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            color=colour,
            linestyle=linestyle,
            linewidth=1.4,
            alpha=0.72,
            zorder=2,
        )
    )


def deployment_tradeoff_figure(
    rows: list[dict],
    *,
    panel_label: str,
    title: str,
    x_normal_fp32: str,
    x_normal_int8: str,
    x_cached_int8: str,
    xlabel: str,
    log_x: bool,
    path: Path,
) -> None:
    """Render one slide-ready float32 → int8 → cached-int8 comparison."""
    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    fig.suptitle(
        f"Edge Window TCN (width 0.25): macro-F1 versus {title.lower()}",
        fontsize=16,
        weight="bold",
        x=0.08,
        y=0.97,
        ha="left",
    )
    fig.text(
        0.08,
        0.925,
        "Held-out configured test split. Follow each colour: float32 normal → int8 normal → int8 cached embeddings.",
        color=MUTED,
        fontsize=9.3,
    )
    for row in rows:
        colour = row["colour"]
        fp32 = (row[x_normal_fp32], row["f1_fp32"])
        int8 = (row[x_normal_int8], row["f1_int8"])
        cached = (row[x_cached_int8], row["f1_int8_cached"])
        transition_arrow(ax, fp32, int8, colour=colour, linestyle="-")
        transition_arrow(ax, int8, cached, colour=colour, linestyle=(0, (2, 2)))
        dot(ax, *fp32, colour=colour, marker="o")
        dot(ax, *int8, colour=colour, marker="s")
        dot(ax, *cached, colour=colour, marker="^")

    add_context_labels(ax, rows, x_normal_fp32, "f1_fp32", log_x=log_x)
    ax.set_title(f"{panel_label}. {title}", loc="left", weight="bold", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0.455, 0.72)
    ax.set_yticks([0.46, 0.52, 0.58, 0.64, 0.70])
    soften_axes(ax)

    context_legend = [
        Line2D([0], [0], marker="o", color=row["colour"], linestyle="", markersize=7, label=row["label"])
        for row in rows
    ]
    form_legend = [
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, linestyle="", markersize=7, label="float32, normal"),
        Line2D([0], [0], marker="s", color=INK, markerfacecolor=INK, linestyle="", markersize=7, label="int8, normal"),
        Line2D([0], [0], marker="^", color=INK, markerfacecolor=INK, linestyle="", markersize=7, label="int8, cached embeddings"),
        Line2D([0], [0], color=INK, linewidth=1.4, label="solid arrow: static int8 quantization"),
        Line2D([0], [0], color=INK, linewidth=1.4, linestyle=(0, (2, 2)), label="dotted arrow: embedding cache"),
    ]
    fig.legend(
        handles=context_legend + form_legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.20)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def deployment_tradeoff_figures(rows: list[dict], assets: Path) -> None:
    """Render individual latency, memory, and model-size figures for slides."""
    specs = [
        ("A", "CPU latency", "latency_fp32_normal", "latency_int8_normal", "latency_int8_cached", "Median inference latency (ms, log scale)", True, "edge-window-tcn-latency-tradeoff.svg"),
        ("B", "Peak activation memory", "memory_fp32_normal", "memory_int8_normal", "memory_int8_cached", "Peak activation memory (KiB, log scale)", True, "edge-window-tcn-memory-tradeoff.svg"),
        ("C", "Exported ONNX size", "size_fp32_kib", "size_int8_kib", "size_int8_cached_kib", "Exported ONNX artifact size (KiB)", False, "edge-window-tcn-model-size-tradeoff.svg"),
    ]
    for panel_label, title, fp32, int8, cached, xlabel, log_x, filename in specs:
        deployment_tradeoff_figure(
            rows,
            panel_label=panel_label,
            title=title,
            x_normal_fp32=fp32,
            x_normal_int8=int8,
            x_cached_int8=cached,
            xlabel=xlabel,
            log_x=log_x,
            path=assets / filename,
        )


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
    new_x = x0 + 14 * (box_w + gap)
    label_box(ax, new_x, 3.25, box_w, 0.72, "new\nCNN", face=FUTURE + "2e", edge=FUTURE, fontsize=7.0)
    ax.add_patch(
        FancyArrowPatch(
            (new_x + box_w / 2, 3.22),
            (new_x + box_w / 2, 2.65),
            arrowstyle="-|>",
            mutation_scale=11,
            color=INK,
            lw=1.1,
        )
    )
    for index, colour in enumerate(colours):
        x = x0 + index * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 1.92), box_w, 0.68, facecolor=colour + "55", edgecolor=colour, linewidth=0.9))
        ax.text(x + box_w / 2, 2.26, "e", ha="center", va="center", fontsize=7.1, color=INK)
    ax.text(7.75, 1.38, "rolling buffer of 15 cached 24-D embeddings", ha="center", fontsize=8.1, color=INK)
    ax.add_patch(FancyArrowPatch((15.2, 2.26), (16.2, 2.26), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
    label_box(ax, 16.3, 1.78, 1.55, 0.96, "TCN\n→ t", face="#edf0ff", edge="#465cc7", fontsize=7.2)
    ax.text(0.0, 0.62, f"1 encoder pass per hop  ·  {future['memory_fp32_cached']:.1f} KiB measured peak  ·  same valid-stream classes", fontsize=8.5, color=INK, weight="bold")

    ax = mosaic["memory"]
    ax.set_title("B. Measured working set: normal versus cached", loc="left", weight="bold", fontsize=12, pad=8)
    categories = ["float32", "int8"]
    normal = [future["memory_fp32_normal"], future["memory_int8_normal"]]
    cached = [future["memory_fp32_cached"], future["memory_int8_cached"]]
    x = [0, 1]
    width = 0.31
    normal_bars = ax.bar([v - width / 2 for v in x], normal, width, color="#aab7c2")
    cached_bars = ax.bar([v + width / 2 for v in x], cached, width, color=future["colour"])
    ax.set_yscale("log")
    ax.set_xticks(x, categories)
    ax.set_ylabel("Peak activation memory (KiB, log scale)")
    ax.yaxis.grid(True, color=GRID, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    for bars, values in ((normal_bars, normal), (cached_bars, cached)):
        for bar, value in zip(bars, values):
            schedule = "normal" if bars is normal_bars else "cached"
            ax.annotate(
                f"{value:.1f}\n{schedule}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.2,
                weight="bold",
            )
    fig.suptitle("Caching embeddings removes repeated CNN work — not the 7 s look-ahead", fontsize=16, weight="bold", x=0.04, ha="left")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    deployment_tradeoff_figures(rows, ASSETS)
    cache_explainer(rows, ASSETS / "edge-window-tcn-embedding-cache-explainer.svg")
    print("Wrote Edge Window TCN deployment summary figures.")


if __name__ == "__main__":
    main()
