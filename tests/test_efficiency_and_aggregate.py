from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from distrimuse_imu_edge.evaluation.aggregate import aggregate_results
from distrimuse_imu_edge.evaluation.efficiency import (
    compute_model_stats,
    compute_streaming_model_stats,
)
from distrimuse_imu_edge.models import build_model


def test_efficiency_report_contains_edge_metrics() -> None:
    model = build_model(
        "edge_window_tcn",
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=7,
    )
    stats = compute_model_stats(model, context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=2)

    assert stats["total_params"] > 0
    assert stats["model_size_mb"] > 0
    assert "gflops" in stats
    assert "cpu_latency_median_ms" in stats
    assert stats["input_shape"] == [1, 8, 20, 6]
    assert stats["energy"]["energy_per_inference_mj"] > 0
    assert stats["energy"]["assumptions"]["name"]


def test_efficiency_energy_uses_hop_and_profile_from_caller() -> None:
    model = build_model(
        "edge_window_tcn",
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=7,
    )
    kwargs: dict = {
        "context_len": 8,
        "window_size_s": 0.2,
        "n_channels": 6,
        "fs": 100,
        "latency_repeats": 2,
    }
    slow_hop = compute_model_stats(
        model, **kwargs, hop_size_s=10.0, energy_profile="nrf54l15_m33_128mhz"
    )
    fast_hop = compute_model_stats(
        model, **kwargs, hop_size_s=1.0, energy_profile="nrf54l15_m33_128mhz"
    )

    # Per-inference energy is a model property and must not move with the hop.
    assert slow_hop["energy"]["energy_per_inference_mj"] == pytest.approx(
        fast_hop["energy"]["energy_per_inference_mj"]
    )
    # Average power is a duty-cycle property and must scale with the hop.
    assert slow_hop["energy"]["avg_power_mw"] < fast_hop["energy"]["avg_power_mw"]
    assert slow_hop["energy"]["est_battery_life_h"] > fast_hop["energy"]["est_battery_life_h"]


def test_compression_label_alone_does_not_grant_int8_energy() -> None:
    """A `dynamic_quant` label on an unquantised model must change nothing.

    The int8 MAC share is measured from the traced layers, so claiming
    compression in the descriptor cannot buy a cheaper energy estimate.
    """
    model = build_model("edge_cnn", n_classes=3, input_channels=6, width_mult=0.25)
    kwargs: dict = {
        "context_len": 1,
        "window_size_s": 0.2,
        "n_channels": 6,
        "fs": 100,
        "latency_repeats": 2,
    }
    plain = compute_model_stats(model, **kwargs, compression=None)
    mislabelled = compute_model_stats(model, **kwargs, compression={"method": "dynamic_quant"})

    assert plain["energy"]["numeric_format"] == "float32"
    assert mislabelled["energy"]["numeric_format"] == "float32"
    assert mislabelled["energy"]["int8_mac_fraction"] == 0.0
    assert mislabelled["energy"]["energy_per_inference_mj"] == pytest.approx(
        plain["energy"]["energy_per_inference_mj"]
    )


def test_int8_mac_fraction_counts_only_quantized_leaf_layers() -> None:
    """The share must come from leaves, and only genuinely-int8 ones.

    torchinfo also reports container modules holding their children's aggregated
    MACs, so summing every entry would double-count.
    """
    from distrimuse_imu_edge.evaluation.efficiency import _int8_mac_fraction

    class _FakeQuantizedConv:
        pass

    _FakeQuantizedConv.__module__ = "torch.ao.nn.quantized.modules.conv"

    class _Layer:
        def __init__(self, module, macs, is_leaf_layer=True):
            self.module = module
            self.macs = macs
            self.is_leaf_layer = is_leaf_layer

    class _Summary:
        def __init__(self, layers):
            self.summary_list = layers

    float_conv = torch.nn.Conv1d(4, 4, 3)
    quant_conv = _FakeQuantizedConv()

    # A container carrying the aggregated total must be ignored.
    summary = _Summary(
        [
            _Layer(torch.nn.Sequential(), 1_000, is_leaf_layer=False),
            _Layer(quant_conv, 750),
            _Layer(float_conv, 250),
            _Layer(torch.nn.ReLU(), 0),
        ]
    )
    assert _int8_mac_fraction(summary) == pytest.approx(0.75)

    assert _int8_mac_fraction(_Summary([_Layer(float_conv, 100)])) == 0.0
    assert _int8_mac_fraction(_Summary([_Layer(quant_conv, 100)])) == 1.0
    # No MAC-bearing leaves must understate rather than invent efficiency.
    assert _int8_mac_fraction(_Summary([_Layer(torch.nn.ReLU(), 0)])) == 0.0
    assert _int8_mac_fraction(object()) == 0.0


def test_peak_activation_bytes_takes_the_largest_layer_not_the_sum() -> None:
    """Peak activation memory is a ping-pong buffer, not a running total.

    Only one layer's input+output tensors need to be resident at a time, so the
    reported figure must be the maximum across layers, not the sum of all of
    them.
    """
    from distrimuse_imu_edge.evaluation.efficiency import _peak_activation_bytes

    class _Layer:
        def __init__(self, input_size, output_size, is_leaf_layer=True):
            self.input_size = input_size
            self.output_size = output_size
            self.is_leaf_layer = is_leaf_layer

    class _Summary:
        def __init__(self, layers):
            self.summary_list = layers

    summary = _Summary(
        [
            _Layer([1, 6, 312], [1, 16, 312], is_leaf_layer=False),  # container, ignored
            _Layer([1, 6, 312], [1, 16, 312]),  # 1872 + 4992 = 6864 elements
            _Layer([1, 16, 312], [1, 16, 312]),  # 4992 + 4992 = 9984 elements, the peak
            _Layer([1, 16, 156], [1, 9]),  # tiny classifier head
        ]
    )
    # The peak layer alone: (4992 + 4992) elements * 4 bytes (float32).
    assert _peak_activation_bytes(summary) == 9984 * 4

    # A shape with a negative dimension (torchinfo's recursive-layer marker)
    # must be skipped rather than crash or corrupt the max.
    negative = _Summary([_Layer([1, 6, 312], [-1, 16, 312])])
    assert _peak_activation_bytes(negative) is None

    # No leaves, or an object with no summary_list, understates rather than
    # invents a figure.
    assert _peak_activation_bytes(_Summary([])) is None
    assert _peak_activation_bytes(object()) is None


def test_compute_model_stats_reports_peak_activation_memory() -> None:
    model = build_model(
        "edge_window_tcn",
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=7,
    )
    stats = compute_model_stats(
        model, context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=2
    )

    assert stats["peak_activation_bytes_fp32"] > 0
    assert stats["peak_activation_kib_fp32"] == pytest.approx(
        stats["peak_activation_bytes_fp32"] / 1024, rel=1e-6
    )
    # The int8 projection is a straight 4x, not an independently traced value.
    assert stats["peak_activation_kib_int8_est"] == pytest.approx(
        stats["peak_activation_kib_fp32"] / 4, rel=1e-6
    )


def test_streaming_stats_peak_memory_is_far_below_batched() -> None:
    """The whole point of streaming: memory should drop, not just latency.

    Batched peak memory scales with the number of context windows (the
    encode-windows reshape puts all of them through the encoder at once);
    streaming only ever encodes one window at a time, so its peak should stay
    close to the current-only figure regardless of context length.
    """
    model = build_model(
        "edge_window_tcn",
        n_classes=9,
        input_channels=6,
        width_mult=0.25,
        current_index=7,
    )
    batched = compute_model_stats(
        model, context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=2
    )
    streaming = compute_streaming_model_stats(
        model, total_context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=2
    )

    assert streaming["peak_activation_kib_fp32_streaming"] > 0
    assert (
        streaming["peak_activation_kib_fp32_streaming"]
        < batched["peak_activation_kib_fp32"]
    )
    # int8 estimate is a straight 4x, same convention as compute_model_stats.
    assert streaming["peak_activation_kib_int8_est_streaming"] == pytest.approx(
        streaming["peak_activation_kib_fp32_streaming"] / 4, rel=1e-6
    )


def test_streaming_stats_peak_memory_does_not_scale_with_context_length() -> None:
    """Streaming's peak memory is ~constant in N; batched grows with N.

    Uses the real training window size (3 s @ 104 Hz, 312 samples) rather than
    a tiny debug window: at realistic scale the encoder's activations (39 KiB,
    see DEPLOYMENT_HARDWARE.md) dwarf the temporal block's (well under 3 KiB
    even at N=15), so the streaming peak is the encoder's alone and genuinely
    independent of N. A tiny debug window inverts that ordering and would make
    this assertion architecture-dependent rather than a real property.
    """
    current_only = build_model(
        "edge_window_tcn", n_classes=9, input_channels=6, width_mult=0.25, current_index=0
    )
    past_and_future = build_model(
        "edge_window_tcn",
        n_classes=9,
        input_channels=6,
        width_mult=0.25,
        current_index=7,
        bidirectional=True,
    )
    kwargs: dict = {"window_size_s": 3.0, "n_channels": 6, "fs": 104, "latency_repeats": 2}
    short = compute_streaming_model_stats(current_only, total_context_len=1, **kwargs)
    long = compute_streaming_model_stats(past_and_future, total_context_len=15, **kwargs)

    # Same encoder architecture (width 0.25) regardless of context length, so
    # the streaming peak — one window through the encoder — must match
    # exactly; only the (tiny, shape-only) temporal-block trace differs.
    assert short["peak_activation_kib_fp32_streaming"] == pytest.approx(
        long["peak_activation_kib_fp32_streaming"], rel=1e-6
    )


def test_streaming_stats_reports_latency() -> None:
    model = build_model(
        "edge_window_tcn", n_classes=9, input_channels=6, width_mult=0.25, current_index=7
    )
    # latency_repeats=2 (as elsewhere in this file) makes the nearest-rank p95
    # index degenerate — with 10 repeats the p95-must-be->=-median relationship
    # is meaningful to assert.
    stats = compute_streaming_model_stats(
        model, total_context_len=8, window_size_s=0.2, n_channels=6, fs=100, latency_repeats=10
    )
    assert stats["cpu_latency_median_ms_streaming"] > 0
    assert stats["cpu_latency_p95_ms_streaming"] >= stats["cpu_latency_median_ms_streaming"]


def test_int8_mac_fraction_override_beats_the_traced_value() -> None:
    """Needed when the deployed artifact is not the traced module.

    An int8 ONNX graph exported from a float32 module traces as all-float32, so
    the caller must be able to supply the share it measured on the real artifact.
    """
    model = build_model("edge_cnn", n_classes=3, input_channels=6, width_mult=0.25)
    kwargs: dict = {
        "context_len": 1,
        "window_size_s": 0.2,
        "n_channels": 6,
        "fs": 100,
        "latency_repeats": 2,
    }
    traced = compute_model_stats(model, **kwargs)
    overridden = compute_model_stats(model, **kwargs, int8_mac_fraction=1.0)

    assert traced["energy"]["numeric_format"] == "float32"
    assert overridden["energy"]["numeric_format"] == "int8"
    assert overridden["energy"]["int8_mac_fraction"] == 1.0
    assert (
        overridden["energy"]["energy_per_inference_mj"]
        < traced["energy"]["energy_per_inference_mj"]
    )


def test_runtime_config_resolves_energy_profile_round_trip(tmp_path) -> None:
    """The profile written to config.resolved.yaml must be re-readable."""
    from distrimuse_imu_edge.cli.common import load_runtime_config
    from distrimuse_imu_edge.evaluation.energy import resolve_profile

    config = tmp_path / "cfg.yaml"
    config.write_text(
        "data:\n  campaign: test\nenergy:\n  profile: stm32l4_m4f_80mhz\n  p_active_mw: 12.5\n",
        encoding="utf-8",
    )

    _, _, resolved = load_runtime_config(config)

    assert resolved["energy"]["name"] == "stm32l4_m4f_80mhz+overrides"
    assert resolved["energy"]["p_active_mw"] == 12.5
    # Feeding the resolved dict back must reproduce the same profile, not raise.
    assert resolve_profile(resolved["energy"]).p_active_mw == 12.5
    assert resolve_profile(resolved["energy"]).name == "stm32l4_m4f_80mhz+overrides"


def test_runtime_config_rejects_bad_energy_profile_at_load_time(tmp_path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  campaign: test\nenergy:\n  profile: nope\n", encoding="utf-8")

    from distrimuse_imu_edge.cli.common import load_runtime_config

    with pytest.raises(KeyError):
        load_runtime_config(config)


def test_aggregate_sorts_by_test_f1(tmp_path) -> None:
    for run, f1, gflops in [("a", 0.4, 0.2), ("b", 0.8, 0.4)]:
        reports = tmp_path / run / "reports"
        reports.mkdir(parents=True)
        (reports / "metrics.json").write_text(json.dumps({"model": run, "test_macro_f1": f1, "val_macro_f1": f1}), encoding="utf-8")
        (reports / "model_stats.json").write_text(json.dumps({"gflops": gflops, "model_size_mb": 1.0, "total_params": 10}), encoding="utf-8")

    df = aggregate_results(tmp_path)

    assert df.iloc[0]["run_name"] == "b"
    assert {"test_macro_f1", "gflops", "model_size_mb", "total_params"}.issubset(df.columns)


def test_aggregate_surfaces_energy_columns_and_tolerates_absent_energy(tmp_path) -> None:
    with_energy = tmp_path / "with_energy" / "reports"
    with_energy.mkdir(parents=True)
    (with_energy / "metrics.json").write_text(
        json.dumps({"model": "edge_cnn", "test_macro_f1": 0.7, "val_macro_f1": 0.7}),
        encoding="utf-8",
    )
    (with_energy / "model_stats.json").write_text(
        json.dumps(
            {
                "gflops": 0.03,
                "model_size_mb": 0.27,
                "total_params": 63081,
                "energy": {
                    "avg_power_mw": 18.45,
                    "est_battery_life_h": 36.59,
                    "duty_cycle": 0.92,
                    "assumptions": {"name": "nrf54l15_m33_128mhz"},
                },
            }
        ),
        encoding="utf-8",
    )

    # A run predating energy reporting must still aggregate, with nulls.
    legacy = tmp_path / "legacy" / "reports"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(
        json.dumps({"model": "edge_tcn", "test_macro_f1": 0.6, "val_macro_f1": 0.6}),
        encoding="utf-8",
    )
    (legacy / "model_stats.json").write_text(
        json.dumps({"gflops": 0.01, "model_size_mb": 0.1, "total_params": 100}),
        encoding="utf-8",
    )

    df = aggregate_results(tmp_path)
    energetic = df[df["run_name"] == "with_energy"].iloc[0]
    old = df[df["run_name"] == "legacy"].iloc[0]

    assert energetic["avg_power_mw"] == 18.45
    assert energetic["est_battery_life_h"] == 36.59
    assert energetic["duty_cycle"] == 0.92
    assert energetic["energy_profile"] == "nrf54l15_m33_128mhz"
    assert pd.isna(old["avg_power_mw"])
    assert pd.isna(old["energy_profile"])


def test_aggregate_empty_columns_match_populated_columns(tmp_path) -> None:
    """The empty-frame schema must not drift from the populated one."""
    populated_dir = tmp_path / "populated"
    reports = populated_dir / "run" / "reports"
    reports.mkdir(parents=True)
    (reports / "metrics.json").write_text(json.dumps({"model": "m", "test_macro_f1": 0.5}), encoding="utf-8")
    (reports / "model_stats.json").write_text(
        json.dumps({"gflops": 0.1, "model_size_mb": 0.5, "total_params": 100}),
        encoding="utf-8",
    )

    populated = aggregate_results(populated_dir)
    empty = aggregate_results(tmp_path / "nothing_here")

    assert set(empty.columns) == set(populated.columns)


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
