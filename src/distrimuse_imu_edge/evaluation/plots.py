from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    import plotly.express as px

    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _short_label(run_name: str) -> str:
    m = re.match(r"^(.+?)_wm([0-9.]+)_ctx\d+(?:_(.+))?$", run_name)
    if not m:
        return run_name
    model, wm, suffix = m.group(1), m.group(2), m.group(3)
    label = f"{model} wm={wm}"
    if suffix:
        label += f" [{suffix}]"
    return label


def write_benchmark_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write comparison HTML plots to output_dir. Returns list of written paths."""
    if not _PLOTLY or df.empty:
        return []

    df = df.copy()
    df["label"] = df["run_name"].map(_short_label)
    df["compression"] = df["compression_method"].fillna("none")
    written: list[Path] = []

    def _save(fig, name: str) -> None:
        p = output_dir / name
        fig.update_layout(height=520, template="plotly_white")
        fig.write_html(str(p), include_plotlyjs="cdn")
        written.append(p)

    # F1 vs GFLOPs
    sub = df.dropna(subset=["test_macro_f1", "gflops"])
    if not sub.empty:
        fig = px.scatter(
            sub,
            x="gflops",
            y="test_macro_f1",
            color="model",
            symbol="compression",
            size="model_size_mb",
            size_max=24,
            hover_name="label",
            hover_data={
                "gflops": ":.4f",
                "model_size_mb": ":.2f",
                "cpu_latency_median_ms": True,
                "run_name": False,
            },
            title="Test macro-F1 vs GFLOPs",
            labels={"gflops": "GFLOPs", "test_macro_f1": "Test macro-F1"},
        )
        _save(fig, "f1_vs_gflops.html")

    # F1 vs CPU latency
    sub = df.dropna(subset=["test_macro_f1", "cpu_latency_median_ms"])
    if not sub.empty:
        fig = px.scatter(
            sub,
            x="cpu_latency_median_ms",
            y="test_macro_f1",
            color="model",
            symbol="compression",
            size="model_size_mb",
            size_max=24,
            hover_name="label",
            hover_data={
                "gflops": ":.4f",
                "model_size_mb": ":.2f",
                "cpu_latency_median_ms": ":.2f",
                "run_name": False,
            },
            title="Test macro-F1 vs CPU inference latency",
            labels={
                "cpu_latency_median_ms": "CPU latency (ms, median)",
                "test_macro_f1": "Test macro-F1",
            },
        )
        _save(fig, "f1_vs_latency.html")

    # F1 vs model size
    sub = df.dropna(subset=["test_macro_f1", "model_size_mb"])
    if not sub.empty:
        fig = px.scatter(
            sub,
            x="model_size_mb",
            y="test_macro_f1",
            color="model",
            symbol="compression",
            hover_name="label",
            hover_data={
                "gflops": ":.4f",
                "cpu_latency_median_ms": True,
                "run_name": False,
            },
            title="Test macro-F1 vs model size",
            labels={"model_size_mb": "Model size (MB)", "test_macro_f1": "Test macro-F1"},
        )
        _save(fig, "f1_vs_size.html")

    # F1 bar chart
    sub = df.dropna(subset=["test_macro_f1"]).sort_values("test_macro_f1", ascending=False)
    if not sub.empty:
        fig = px.bar(
            sub,
            x="label",
            y="test_macro_f1",
            color="model",
            pattern_shape="compression",
            hover_data={
                "gflops": ":.4f",
                "model_size_mb": ":.2f",
                "cpu_latency_median_ms": True,
            },
            title="Test macro-F1 by model / compression",
            labels={"label": "", "test_macro_f1": "Test macro-F1"},
        )
        fig.update_layout(xaxis_tickangle=35)
        _save(fig, "f1_bar.html")

    return written
