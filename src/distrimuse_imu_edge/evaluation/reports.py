from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from distrimuse_imu_edge.evaluation.artifacts import write_test_prediction_artifacts
from distrimuse_imu_edge.evaluation.metrics import class_names_for


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
    data_config = resolved_config.get("data", {}) or {}
    probability_columns = [column for column in predictions if column.startswith("prob_")]
    inferred_n_classes = len(probability_columns)
    if not inferred_n_classes and not predictions.empty:
        inferred_n_classes = int(predictions[["y_true", "y_pred"]].to_numpy().max()) + 1
    n_classes = int(data_config.get("n_classes", inferred_n_classes))
    write_test_prediction_artifacts(
        output_dir=output_dir,
        predictions=predictions,
        class_names=class_names_for(n_classes, task_col=str(data_config.get("task_col", "big_movement"))),
        hop_size_s=float(data_config.get("hop_size_s", 1.0)),
    )
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
    timeline_links = [
        f"<li><a href='{item.name}'>{item.stem.replace('_', ' ')}</a></li>"
        for item in sorted(path.parent.glob("prediction_timeline_subject_*.html"))
    ]
    confusion_links = [
        f"<li><a href='../confusion_matrices/{item.name}'>{item.stem.replace('_', ' ')}</a></li>"
        for item in sorted((path.parent.parent / "confusion_matrices").glob("*.html"))
    ]
    path.write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'><title>IMU Edge Run</title>",
                "<h1>IMU Edge Run</h1>",
                f"<p>{val_label}: {val}</p>",
                f"<p>{test_label}: {test}</p>",
                f"<p>GFLOPs: {gflops}</p>",
                f"<p>model size MB: {size}</p>",
                "<h2>Confusion matrices</h2>",
                f"<ul>{''.join(confusion_links)}</ul>",
                "<h2>Prediction timelines</h2>",
                f"<ul>{''.join(timeline_links)}</ul>",
            ]
        ),
        encoding="utf-8",
    )
