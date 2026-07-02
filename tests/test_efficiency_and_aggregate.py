from __future__ import annotations

import json

import pandas as pd

from distrimuse_imu_edge.evaluation.aggregate import aggregate_results
from distrimuse_imu_edge.evaluation.efficiency import compute_model_stats
from distrimuse_imu_edge.models import build_model


def test_efficiency_report_contains_edge_metrics() -> None:
    model = build_model("edge_cnn", n_classes=3, input_channels=6, width_mult=0.25)
    stats = compute_model_stats(model, context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=2)

    assert stats["total_params"] > 0
    assert stats["model_size_mb"] > 0
    assert "gflops" in stats
    assert "cpu_latency_median_ms" in stats
    assert stats["input_shape"] == [1, 8, 20, 6]


def test_aggregate_sorts_by_test_f1(tmp_path) -> None:
    for run, f1, gflops in [("a", 0.4, 0.2), ("b", 0.8, 0.4)]:
        reports = tmp_path / run / "reports"
        reports.mkdir(parents=True)
        (reports / "metrics.json").write_text(json.dumps({"model": run, "test_macro_f1": f1, "val_macro_f1": f1}), encoding="utf-8")
        (reports / "model_stats.json").write_text(json.dumps({"gflops": gflops, "model_size_mb": 1.0, "total_params": 10}), encoding="utf-8")

    df = aggregate_results(tmp_path)

    assert df.iloc[0]["run_name"] == "b"
    assert {"test_macro_f1", "gflops", "model_size_mb", "total_params"}.issubset(df.columns)


def test_aggregate_keeps_wisdm_pretraining_metrics_separate(tmp_path) -> None:
    finetune_reports = tmp_path / "finetune" / "reports"
    finetune_reports.mkdir(parents=True)
    (finetune_reports / "metrics.json").write_text(
        json.dumps({"model": "edge_cnn", "test_macro_f1": 0.5, "val_macro_f1": 0.4}),
        encoding="utf-8",
    )
    (finetune_reports / "model_stats.json").write_text(
        json.dumps({"gflops": 0.1, "model_size_mb": 1.0, "total_params": 10}),
        encoding="utf-8",
    )

    pretrain_reports = tmp_path / "edge_cnn_wisdm19_pretrain" / "reports"
    pretrain_reports.mkdir(parents=True)
    (pretrain_reports / "metrics.json").write_text(
        json.dumps(
            {
                "model": "edge_cnn",
                "dataset": "wisdm19",
                "wisdm_test_macro_f1": 0.9,
                "wisdm_val_macro_f1": 0.8,
            }
        ),
        encoding="utf-8",
    )
    (pretrain_reports / "model_stats.json").write_text(
        json.dumps({"gflops": 0.1, "model_size_mb": 1.0, "total_params": 10}),
        encoding="utf-8",
    )

    df = aggregate_results(tmp_path)
    wisdm = df[df["run_name"] == "edge_cnn_wisdm19_pretrain"].iloc[0]

    assert df.iloc[0]["run_name"] == "finetune"
    assert pd.isna(wisdm["test_macro_f1"])
    assert wisdm["wisdm_test_macro_f1"] == 0.9
