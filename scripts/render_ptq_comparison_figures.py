"""Render the float32-versus-int8 PTQ figures for the context-comparison report.

Reads the ``quantization_comparison.json`` written by ``imu-edge-quantize`` for
each context mode, so the figures cannot drift from the recorded results. Run
from the repository root::

    uv run python scripts/render_ptq_comparison_figures.py

Palette
-------
Context mode carries hue and precision carries texture, so no new hue is
introduced for float32-versus-int8 and the encoding survives greyscale print and
colour-vision deficiency.

The hues are ``#2a78d6`` / ``#eb6834`` / ``#1baf7a``, replacing the matplotlib
tab10 defaults the report's older figures use. tab10's green and orange
(``#2ca02c`` / ``#ff7f0e``) sit at OKLab CVD ΔE 0.7 under protanopia — a
protanope cannot tell the "past 7 + current" series from the "past 7 + current +
future 7" series at all. The replacements clear every gate on both the adjacent
and all-pairs checks (worst CVD ΔE 9.2, worst normal-vision ΔE 24.0), while
keeping blue for current-only and orange for past-context so the reader's mapping
from the earlier figures still holds.

``#1baf7a`` falls below 3:1 contrast on a light surface, so every bar carries a
direct value label and each figure is mirrored by a table in the report.

Bars keep matplotlib's square ends rather than rounded data-ends, to stay
consistent with the fourteen existing figures in the same report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = Path("experiments/results")
ASSETS = RESULTS / "edge_window_tcn_context_report_assets"

# (label, run directory) in the report's established narrative order.
RUNS = [
    ("Current only", "edge_window_tcn_wm025_current_int8"),
    ("Past 7 + current", "edge_window_tcn_wm025_past7_current_int8"),
    ("Past 7 + current\n+ future 7", "edge_window_tcn_wm025_centered_scratch_int8"),
]
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


def load() -> list[dict]:
    out = []
    for label, run in RUNS:
        path = RESULTS / run / "reports" / "quantization_comparison.json"
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run imu-edge-quantize for every context mode first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append({"label": label, "run": run, **payload})
    return out


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


def _precision_legend(ax, **kwargs) -> None:
    """Legend for the texture channel only — hue is labelled by the axis ticks."""
    ax.legend(
        handles=[
            Patch(facecolor="#8a8a8a", edgecolor="none", label="float32"),
            Patch(
                facecolor="#8a8a8a",
                edgecolor=SURFACE,
                hatch=INT8_HATCH,
                label="int8 (PTQ)",
            ),
        ],
        frameon=False,
        fontsize=9,
        **kwargs,
    )


def figure_headline(rows: list[dict], path: Path) -> None:
    """Accuracy cost and size benefit, one panel each."""
    fig, (ax_f1, ax_size) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = range(len(rows))
    # A 2px surface gap between the paired bars.
    width = 0.36
    offset = width / 2 + 0.02

    for index, row in enumerate(rows):
        colour = SERIES_COLORS[index]
        f32 = row["test_macro_f1"]["float32"]
        i8 = row["test_macro_f1"]["int8"]
        ax_f1.bar(index - offset, f32, width, color=colour, linewidth=0)
        ax_f1.bar(
            index + offset,
            i8,
            width,
            color=colour,
            edgecolor=SURFACE,
            hatch=INT8_HATCH,
            linewidth=0,
        )
        ax_f1.text(index - offset, f32 + 0.008, f"{f32:.3f}", ha="center", fontsize=9)
        ax_f1.text(index + offset, i8 + 0.008, f"{i8:.3f}", ha="center", fontsize=9)
        delta = row["test_macro_f1"]["delta"]
        ax_f1.text(
            index,
            max(f32, i8) + 0.055,
            f"{delta:+.4f}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK if delta < -0.01 else INK_SOFT,
        )

    ax_f1.set_ylabel("Test macro-F1")
    ax_f1.set_title(
        "Accuracy cost of int8 post-training quantization", fontsize=11, pad=14
    )
    ax_f1.set_ylim(0, 0.85)
    ax_f1.set_xticks(list(x))
    ax_f1.set_xticklabels([r["label"] for r in rows], fontsize=9)
    _recessive_axes(ax_f1)
    _precision_legend(ax_f1, loc="upper left")

    for index, row in enumerate(rows):
        colour = SERIES_COLORS[index]
        comp = row["compression"]
        f32 = comp["onnx_fp32_kib"]
        i8 = comp["onnx_int8_kib"]
        ax_size.bar(index - offset, f32, width, color=colour, linewidth=0)
        ax_size.bar(
            index + offset,
            i8,
            width,
            color=colour,
            edgecolor=SURFACE,
            hatch=INT8_HATCH,
            linewidth=0,
        )
        ax_size.text(index - offset, f32 + 1.2, f"{f32:.0f}", ha="center", fontsize=9)
        ax_size.text(index + offset, i8 + 1.2, f"{i8:.0f}", ha="center", fontsize=9)
        pct = 100.0 * (i8 - f32) / f32
        ax_size.text(
            index,
            max(f32, i8) + 8.0,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
        )

    ax_size.set_ylabel("ONNX artifact size (KiB)")
    ax_size.set_title("Deployed artifact size", fontsize=11, pad=14)
    ax_size.set_ylim(0, 125)
    ax_size.set_xticks(list(x))
    ax_size.set_xticklabels([r["label"] for r in rows], fontsize=9)
    _recessive_axes(ax_size)
    _precision_legend(ax_size, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_per_class(rows: list[dict], path: Path) -> None:
    """Where the accuracy went: per-class F1 change, signed."""
    classes = list(rows[0]["test_per_class_f1"].keys())
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    n = len(rows)
    height = 0.24
    positions = range(len(classes))

    for index, row in enumerate(rows):
        deltas = [row["test_per_class_f1"][c]["delta"] for c in classes]
        # Centre the group on each class tick, 2px-equivalent gaps between bars.
        shift = (index - (n - 1) / 2) * (height + 0.02)
        ax.barh(
            [p + shift for p in positions],
            deltas,
            height,
            color=SERIES_COLORS[index],
            edgecolor="none",
            linewidth=0,
            label=row["label"].replace("\n", " "),
        )

    ax.axvline(0.0, color=INK_SOFT, linewidth=1.0, zorder=3)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class test F1 change, int8 − float32")
    ax.set_title(
        "Quantization damage by class: the causal models absorb it,\n"
        "the future-context model does not",
        fontsize=11,
        pad=14,
    )
    _recessive_axes(ax, y_grid=False, x_grid=True)
    # Every bar is negative, so it runs leftward into any in-axes legend. Park the
    # legend below the axis label instead of overlapping Falling and Hand.
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        title="Context mode",
    )
    ax.get_legend().get_title().set_fontsize(9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load()
    ASSETS.mkdir(parents=True, exist_ok=True)
    headline = ASSETS / "ptq-float32-vs-int8.svg"
    per_class = ASSETS / "ptq-per-class-delta.svg"
    figure_headline(rows, headline)
    figure_per_class(rows, per_class)
    for path in (headline, per_class):
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
