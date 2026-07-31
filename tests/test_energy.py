from __future__ import annotations

import pytest

from distrimuse_imu_edge.evaluation.energy import (
    DEFAULT_PROFILE_NAME,
    ENERGY_PROFILES,
    EnergyProfile,
    describe_numeric_format,
    estimate_energy,
    resolve_profile,
)


def _profile(**overrides) -> EnergyProfile:
    """A profile with round numbers so expected values are hand-checkable."""
    base = EnergyProfile(
        name="test",
        f_clock_hz=100e6,
        macs_per_cycle_int8=2.0,
        macs_per_cycle_float32=0.5,
        p_active_mw=10.0,
        p_sleep_mw=0.0,
        battery_capacity_mah=100.0,
        battery_voltage_v=3.0,
    )
    return base if not overrides else type(base)(**{**base.to_dict(), **overrides})


def test_energy_matches_hand_computed_duty_cycle() -> None:
    # 100M MACs at 2 MACs/cycle on a 100 MHz core = 0.5 s active.
    # 0.5 s at 10 mW = 5 mJ. Hop 1 s and zero sleep power -> 5 mW average.
    # 100 mAh at 3.0 V = 300 mWh; 300 / 5 = 60 h.
    result = estimate_energy(
        macs=100_000_000,
        hop_size_s=1.0,
        profile=_profile(),
        int8_mac_fraction=1.0,
    )

    assert result["active_time_per_inference_ms"] == pytest.approx(500.0)
    assert result["energy_per_inference_mj"] == pytest.approx(5.0)
    assert result["avg_power_mw"] == pytest.approx(5.0)
    assert result["duty_cycle"] == pytest.approx(0.5)
    assert result["est_battery_life_h"] == pytest.approx(60.0)
    assert result["est_battery_life_days"] == pytest.approx(2.5)
    assert result["real_time_feasible"] is True


def test_sleep_power_is_duty_cycle_weighted_not_added_flat() -> None:
    # duty = 0.5, so average = 10*0.5 + 1*0.5 = 5.5 mW.
    # A flat "energy/hop + p_sleep" would give 6.0 mW and overcount sleep
    # during the active window.
    result = estimate_energy(
        macs=100_000_000,
        hop_size_s=1.0,
        profile=_profile(p_sleep_mw=1.0),
        int8_mac_fraction=1.0,
    )

    assert result["avg_power_mw"] == pytest.approx(5.5)


def test_float32_costs_more_than_int8_on_same_profile() -> None:
    kwargs = {"macs": 10_000_000, "hop_size_s": 1.0, "profile": _profile()}
    int8 = estimate_energy(**kwargs, int8_mac_fraction=1.0)
    float32 = estimate_energy(**kwargs, int8_mac_fraction=0.0)

    # 2.0 vs 0.5 MACs/cycle -> exactly 4x the active time and energy.
    assert float32["energy_per_inference_mj"] == pytest.approx(
        4 * int8["energy_per_inference_mj"]
    )
    assert float32["numeric_format"] == "float32"
    assert int8["numeric_format"] == "int8"


def test_partial_quantization_is_weighted_by_mac_share_not_treated_as_int8() -> None:
    """A nominally-quantised conv model must not be credited full int8 speed.

    This is the failure mode of inferring the format from a compression label:
    dynamic quantisation converts Linear/GRU but not Conv1d, so a conv-dominated
    model can be labelled "quantised" while ~0% of its MACs run in int8.
    """
    kwargs = {"macs": 10_000_000, "hop_size_s": 10.0, "profile": _profile()}
    all_float = estimate_energy(**kwargs, int8_mac_fraction=0.0)
    all_int8 = estimate_energy(**kwargs, int8_mac_fraction=1.0)
    # What dynamic quant actually achieves on edge_window_tcn: ~0.02% of MACs.
    barely = estimate_energy(**kwargs, int8_mac_fraction=0.0002)

    assert barely["energy_per_inference_mj"] == pytest.approx(
        all_float["energy_per_inference_mj"], rel=1e-3
    )
    assert barely["energy_per_inference_mj"] > 3.9 * all_int8["energy_per_inference_mj"]
    assert barely["numeric_format"] == "mixed (0.0% int8)"
    assert barely["int8_mac_fraction"] == 0.0002


def test_half_quantized_time_is_sum_of_both_paths() -> None:
    # 5M MACs at 2/cycle + 5M at 0.5/cycle on 100 MHz
    #   = 0.025 s + 0.1 s = 0.125 s active.
    result = estimate_energy(
        macs=10_000_000, hop_size_s=1.0, profile=_profile(), int8_mac_fraction=0.5
    )

    assert result["active_time_per_inference_ms"] == pytest.approx(125.0)
    assert result["assumptions"]["macs_per_cycle_used"] == pytest.approx(0.8)


def test_estimate_energy_rejects_out_of_range_int8_fraction() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            estimate_energy(macs=1000, profile=_profile(), int8_mac_fraction=bad)


def test_describe_numeric_format_labels() -> None:
    assert describe_numeric_format(0.0) == "float32"
    assert describe_numeric_format(1.0) == "int8"
    assert describe_numeric_format(0.25) == "mixed (25.0% int8)"


def test_infeasible_duty_cycle_is_flagged_and_power_saturates() -> None:
    # 1G MACs at 2 MACs/cycle on 100 MHz = 5 s active, but the hop is 1 s.
    result = estimate_energy(
        macs=1_000_000_000,
        hop_size_s=1.0,
        profile=_profile(),
        int8_mac_fraction=1.0,
    )

    assert result["real_time_feasible"] is False
    assert result["duty_cycle"] == pytest.approx(5.0)
    assert result["avg_power_mw"] == pytest.approx(10.0)  # never sleeps


def test_energy_is_proportional_to_macs() -> None:
    """Documents the key limitation: ranking is identical to ranking by MACs."""
    small = estimate_energy(macs=1_000_000, profile=_profile(), int8_mac_fraction=1.0)
    large = estimate_energy(macs=8_000_000, profile=_profile(), int8_mac_fraction=1.0)

    assert large["energy_per_inference_mj"] == pytest.approx(
        8 * small["energy_per_inference_mj"]
    )


def test_missing_macs_yields_null_metrics_but_keeps_assumptions() -> None:
    result = estimate_energy(macs=None, profile=_profile())

    assert result["energy_per_inference_mj"] is None
    assert result["avg_power_mw"] is None
    assert result["real_time_feasible"] is None
    assert result["assumptions"]["p_active_mw"] == 10.0


def test_resolve_profile_accepts_name_mapping_and_none() -> None:
    assert resolve_profile(None).name == DEFAULT_PROFILE_NAME
    assert resolve_profile("stm32l4_m4f_80mhz").name == "stm32l4_m4f_80mhz"

    overridden = resolve_profile({"profile": "stm32l4_m4f_80mhz", "p_active_mw": 12.5})
    assert overridden.p_active_mw == 12.5
    assert overridden.f_clock_hz == ENERGY_PROFILES["stm32l4_m4f_80mhz"].f_clock_hz
    # Must not masquerade as a stock profile once values are changed.
    assert overridden.name == "stm32l4_m4f_80mhz+overrides"


def test_resolve_profile_rejects_unknown_names_and_fields() -> None:
    with pytest.raises(KeyError):
        resolve_profile("not_a_real_chip")
    with pytest.raises(ValueError):
        resolve_profile({"profile": DEFAULT_PROFILE_NAME, "p_activ_mw": 10.0})


def test_estimate_energy_rejects_non_positive_hop() -> None:
    with pytest.raises(ValueError):
        estimate_energy(macs=1000, hop_size_s=0.0, profile=_profile())


def test_bundled_profiles_are_self_consistent() -> None:
    for name, profile in ENERGY_PROFILES.items():
        assert profile.name == name, "registry key must match profile name"
        assert profile.f_clock_hz > 0
        assert profile.macs_per_cycle_int8 >= profile.macs_per_cycle_float32
        assert profile.p_active_mw > profile.p_sleep_mw >= 0
        assert profile.notes, "every profile must document where its numbers came from"
