from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def write_run_reports(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    model_stats: dict[str, Any],
    predictions: pd.DataFrame,
    resolved_config: dict[str, Any],
) -> None:
    reports = output_dir / "reports"
    plots = output_dir / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports / "model_stats.json").write_text(json.dumps(model_stats, indent=2), encoding="utf-8")
    predictions.to_parquet(reports / "predictions.parquet", index=False)
    (reports / "config.resolved.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    _write_minimal_plot_index(plots / "index.html", metrics=metrics, stats=model_stats)


def _write_minimal_plot_index(path: Path, *, metrics: dict[str, Any], stats: dict[str, Any]) -> None:
    val_label = "val macro F1"
    test_label = "test macro F1"
    val = metrics.get("val_macro_f1")
    test = metrics.get("test_macro_f1")
    if val is None and "wisdm_val_macro_f1" in metrics:
        val_label = "WISDM val macro F1"
        val = metrics.get("wisdm_val_macro_f1")
    if test is None and "wisdm_test_macro_f1" in metrics:
        test_label = "WISDM test macro F1"
        test = metrics.get("wisdm_test_macro_f1")
    gflops = stats.get("gflops")
    size = stats.get("model_size_mb")
    path.write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'><title>IMU Edge Run</title>",
                "<h1>IMU Edge Run</h1>",
                f"<p>{val_label}: {val}</p>",
                f"<p>{test_label}: {test}</p>",
                f"<p>GFLOPs: {gflops}</p>",
                f"<p>model size MB: {size}</p>",
            ]
        ),
        encoding="utf-8",
    )
