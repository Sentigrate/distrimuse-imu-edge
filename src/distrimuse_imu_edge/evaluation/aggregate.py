from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _dataset_from_config(run_dir: Path) -> str | None:
    config_path = run_dir / "reports" / "config.resolved.yaml"
    if not config_path.exists():
        return None
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    data = config.get("data", {}) or {}
    return data.get("campaign")


def _is_wisdm_pretrain(metrics: dict[str, Any], run_dir: Path) -> bool:
    dataset = metrics.get("dataset") or _dataset_from_config(run_dir)
    return dataset == "wisdm19" or "wisdm_test_macro_f1" in metrics


def aggregate_results(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(results_dir)
    for metrics_path in root.glob("*/reports/metrics.json"):
        run_dir = metrics_path.parents[1]
        stats_path = run_dir / "reports" / "model_stats.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        is_wisdm = _is_wisdm_pretrain(metrics, run_dir)
        # Analytic estimate under the run's declared hardware profile; absent
        # for runs written before energy reporting existed.
        energy = stats.get("energy") or {}
        rows.append(
            {
                "run_name": run_dir.name,
                "model": metrics.get("model"),
                "val_macro_f1": None if is_wisdm else metrics.get("val_macro_f1"),
                "test_macro_f1": None if is_wisdm else metrics.get("test_macro_f1"),
                "wisdm_val_macro_f1": metrics.get("wisdm_val_macro_f1")
                or (metrics.get("val_macro_f1") if is_wisdm else None),
                "wisdm_test_macro_f1": metrics.get("wisdm_test_macro_f1")
                or (metrics.get("test_macro_f1") if is_wisdm else None),
                "context_len": stats.get("context_len"),
                "future_context_len": stats.get("future_context_len", 0),
                "total_context_len": stats.get("total_context_len", stats.get("context_len")),
                "gflops": stats.get("gflops"),
                "macs": stats.get("macs"),
                "model_size_mb": stats.get("model_size_mb"),
                "total_params": stats.get("total_params"),
                "cpu_latency_median_ms": stats.get("cpu_latency_median_ms"),
                "avg_power_mw": energy.get("avg_power_mw"),
                "est_battery_life_h": energy.get("est_battery_life_h"),
                "duty_cycle": energy.get("duty_cycle"),
                "energy_profile": (energy.get("assumptions") or {}).get("name"),
                "compression_method": (stats.get("compression") or {}).get("method"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "run_name",
                "model",
                "val_macro_f1",
                "test_macro_f1",
                "wisdm_val_macro_f1",
                "wisdm_test_macro_f1",
                "context_len",
                "future_context_len",
                "total_context_len",
                "gflops",
                "macs",
                "model_size_mb",
                "total_params",
                "cpu_latency_median_ms",
                "avg_power_mw",
                "est_battery_life_h",
                "duty_cycle",
                "energy_profile",
                "compression_method",
            ]
        )
    df = df.assign(
        _sort_f1=df["test_macro_f1"].fillna(df["wisdm_test_macro_f1"]),
        _sort_is_wisdm=df["test_macro_f1"].isna() & df["wisdm_test_macro_f1"].notna(),
    )
    df = df.sort_values(
        ["_sort_is_wisdm", "_sort_f1", "gflops"],
        ascending=[True, False, True],
        na_position="last",
    ).drop(columns=["_sort_f1", "_sort_is_wisdm"])
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "benchmark_summary.csv", index=False)

    from distrimuse_imu_edge.evaluation.plots import write_benchmark_plots
    write_benchmark_plots(df, root)

    return df
