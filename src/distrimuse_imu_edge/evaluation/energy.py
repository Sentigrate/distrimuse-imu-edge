"""Analytic energy estimates for edge deployment scenarios.

This module turns a MAC count into interpretable units — millijoules per
inference, average milliwatts, and estimated battery life — under an
**explicitly declared hardware assumption**. It is a scenario calculator, not
a measurement.

What the model does
-------------------
The estimate is the standard duty-cycle energy model used throughout embedded
and wireless-sensor-network engineering::

    t_active = MACs / (macs_per_cycle * f_clock)
    E_inference = P_active * t_active
    P_avg = P_active * duty + P_sleep * (1 - duty)      duty = t_active / hop
    battery_life = capacity_mAh * V / P_avg

What the model does NOT do
--------------------------
Three limitations matter enough to state up front, because they determine how
the output may honestly be used.

1. **It is monotonic in MACs.** Every other term is a constant across models,
   so ``energy_per_inference_mj`` is exactly proportional to ``macs`` and adds
   no model-ranking information beyond the existing ``gmacs`` field. The value
   here is unit translation — "6 days on a coin cell" instead of "0.0295
   GMACs" — not a new comparison axis.

2. **It ignores memory movement.** ``P_active`` is treated as one constant, so
   a memory-bound model and a compute-bound model with equal MACs get equal
   energy. On real microcontrollers data movement often dominates arithmetic
   (Horowitz, ISSCC 2014: ~0.2 pJ for an int8 multiply-add versus ~5 pJ for an
   SRAM read). Models that account for this — Yang, Chen & Sze's energy-aware
   pruning tool (CVPR 2017), Accelergy/Timeloop — weight per-level memory
   accesses explicitly. This module does not.

3. **It covers inference only.** Continuous IMU sampling at 104 Hz, sensor
   front-end power, and any radio traffic are excluded, and on a real wearable
   those terms are frequently larger than inference. Do not read
   ``est_battery_life_h`` as a device-level battery projection.

For a number that survives review, measure on the target device with a power
analyser (Joulescope, Otii Arc, Nordic PPK2) or use the MLPerf Tiny energy
harness. This module exists to make the compute term legible, not to replace
that measurement.

Profile values
--------------
The bundled profiles are documented order-of-magnitude figures derived from
typical datasheet run/sleep currents for each part class, not measurements on
a specific board. Each profile records its reasoning in ``notes``. Treat them
as starting points and override them in the config once you have measured your
own hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

# One inference per hop. Mirrors DataConfig.hop_size_s so the module stays
# usable standalone without importing the data package.
DEFAULT_HOP_SIZE_S = 1.0

INT8 = "int8"
FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class EnergyProfile:
    """A complete deployment scenario: a compute part plus a power source.

    Compute and battery live in one object because they are only ever varied
    together — "this model on that device with that cell" is the unit a
    stakeholder reasons about.

    Attributes:
        name: Identifier echoed into ``model_stats.json``.
        f_clock_hz: Clock the inference runs at.
        macs_per_cycle_int8: Sustained multiply-accumulates per cycle for
            quantised int8 kernels. On Cortex-M4/M33 this reflects the dual-MAC
            SIMD instructions CMSIS-NN uses, discounted for loop and
            data-marshalling overhead.
        macs_per_cycle_float32: Sustained MACs per cycle for float32 kernels.
            Typically ~4x worse than int8 on a scalar FPU, which is why the
            numeric format is selected explicitly rather than assumed.
        p_active_mw: Board power while computing, at the profile's supply
            voltage.
        p_sleep_mw: Board power between inferences, assuming RAM retention and
            a low-power timer running.
        battery_capacity_mah: Nominal cell capacity.
        battery_voltage_v: Nominal cell voltage.
        notes: Where the numbers came from, so a reader can judge them.
    """

    name: str
    f_clock_hz: float
    macs_per_cycle_int8: float
    macs_per_cycle_float32: float
    p_active_mw: float
    p_sleep_mw: float
    battery_capacity_mah: float
    battery_voltage_v: float
    notes: str = ""

    def macs_per_cycle(self, numeric_format: str) -> float:
        if numeric_format == INT8:
            return self.macs_per_cycle_int8
        if numeric_format == FLOAT32:
            return self.macs_per_cycle_float32
        raise ValueError(
            f"unknown numeric_format {numeric_format!r}; expected {INT8!r} or {FLOAT32!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "f_clock_hz": self.f_clock_hz,
            "macs_per_cycle_int8": self.macs_per_cycle_int8,
            "macs_per_cycle_float32": self.macs_per_cycle_float32,
            "p_active_mw": self.p_active_mw,
            "p_sleep_mw": self.p_sleep_mw,
            "battery_capacity_mah": self.battery_capacity_mah,
            "battery_voltage_v": self.battery_voltage_v,
            "notes": self.notes,
        }


ENERGY_PROFILES: dict[str, EnergyProfile] = {
    "nrf52840_m4f_64mhz": EnergyProfile(
        name="nrf52840_m4f_64mhz",
        f_clock_hz=64e6,
        macs_per_cycle_int8=2.0,
        macs_per_cycle_float32=0.5,
        p_active_mw=20.0,
        p_sleep_mw=0.01,
        battery_capacity_mah=225.0,
        battery_voltage_v=3.0,
        notes=(
            "Cortex-M4F at 64 MHz, CPU running from flash: ~6.5 mA at 3.0 V. "
            "Sleep assumes System ON with RAM retention and an RTC, ~3 uA. "
            "Battery is a CR2032 coin cell. Order-of-magnitude datasheet-class "
            "figures, not board measurements."
        ),
    ),
    "stm32l4_m4f_80mhz": EnergyProfile(
        name="stm32l4_m4f_80mhz",
        f_clock_hz=80e6,
        macs_per_cycle_int8=2.0,
        macs_per_cycle_float32=0.5,
        p_active_mw=30.0,
        p_sleep_mw=0.01,
        battery_capacity_mah=225.0,
        battery_voltage_v=3.0,
        notes=(
            "Cortex-M4F at 80 MHz in the higher-performance voltage range: "
            "~10 mA at 3.0 V. Sleep assumes a stop mode with RAM retention. "
            "Battery is a CR2032 coin cell."
        ),
    ),
    "stm32u5_m33_160mhz": EnergyProfile(
        name="stm32u5_m33_160mhz",
        f_clock_hz=160e6,
        macs_per_cycle_int8=2.0,
        macs_per_cycle_float32=0.5,
        p_active_mw=55.0,
        p_sleep_mw=0.01,
        battery_capacity_mah=225.0,
        battery_voltage_v=3.0,
        notes=(
            "Cortex-M33 at 160 MHz: ~18 mA at 3.0 V. No Helium vector unit, so "
            "per-cycle throughput matches the M4F class; the gain is clock rate. "
            "Battery is a CR2032 coin cell."
        ),
    ),
    "ethos_u55_64_200mhz": EnergyProfile(
        name="ethos_u55_64_200mhz",
        f_clock_hz=200e6,
        macs_per_cycle_int8=32.0,
        macs_per_cycle_float32=0.5,
        p_active_mw=25.0,
        p_sleep_mw=0.01,
        battery_capacity_mah=225.0,
        battery_voltage_v=3.0,
        notes=(
            "Cortex-M55 with an Ethos-U55-64 NPU at 200 MHz. The NPU is rated "
            "64 int8 MACs/cycle; 32 assumes ~50% utilisation on small 1D "
            "convolutions. Float32 falls back to the scalar CPU path, which is "
            "why the float column is unchanged. Included as a contrast case: it "
            "shows how much of the cost is the absence of an accelerator."
        ),
    ),
}

DEFAULT_PROFILE_NAME = "nrf52840_m4f_64mhz"

_NUMERIC_FIELDS = frozenset(
    {
        "f_clock_hz",
        "macs_per_cycle_int8",
        "macs_per_cycle_float32",
        "p_active_mw",
        "p_sleep_mw",
        "battery_capacity_mah",
        "battery_voltage_v",
    }
)
_OVERRIDABLE = _NUMERIC_FIELDS | {"notes"}


def resolve_profile(spec: EnergyProfile | str | Mapping[str, Any] | None) -> EnergyProfile:
    """Resolve a config value into an ``EnergyProfile``.

    Accepts, in order of precedence:

    * an already-built ``EnergyProfile``;
    * a bundled profile name;
    * a **complete** mapping carrying ``name`` plus every numeric field — the
      round-trip form produced by ``EnergyProfile.to_dict()``, so a resolved
      profile written into ``config.resolved.yaml`` can be read straight back
      even when its name is not a registry key;
    * a **partial** mapping of the form
      ``{"profile": "<name>", "p_active_mw": 12.0}``, where the listed fields
      override the named base;
    * ``None``, which yields the default profile.

    A partial mapping that overrides anything is renamed to
    ``"<base>+overrides"`` so the reported profile name can never claim to be a
    stock profile it isn't.

    Raises:
        KeyError: If a named profile does not exist.
        ValueError: If a mapping contains keys that are not overridable fields.
    """
    if spec is None:
        return ENERGY_PROFILES[DEFAULT_PROFILE_NAME]
    if isinstance(spec, EnergyProfile):
        return spec
    if isinstance(spec, str):
        if spec not in ENERGY_PROFILES:
            raise KeyError(
                f"unknown energy profile {spec!r}; available: {sorted(ENERGY_PROFILES)}"
            )
        return ENERGY_PROFILES[spec]

    payload = dict(spec)
    if "name" in payload and _NUMERIC_FIELDS.issubset(payload):
        extra = set(payload) - _OVERRIDABLE - {"name"}
        if extra:
            raise ValueError(f"unknown energy profile field(s) {sorted(extra)}")
        return EnergyProfile(
            name=str(payload["name"]),
            notes=str(payload.get("notes", "")),
            **{key: float(payload[key]) for key in _NUMERIC_FIELDS},
        )

    base_name = str(payload.pop("profile", DEFAULT_PROFILE_NAME))
    if base_name not in ENERGY_PROFILES:
        raise KeyError(
            f"unknown energy profile {base_name!r}; available: {sorted(ENERGY_PROFILES)}"
        )
    base = ENERGY_PROFILES[base_name]
    unknown = set(payload) - _OVERRIDABLE
    if unknown:
        raise ValueError(
            f"unknown energy profile override(s) {sorted(unknown)}; "
            f"overridable: {sorted(_OVERRIDABLE)}"
        )
    if not payload:
        return base
    overrides: dict[str, Any] = {
        key: (value if key == "notes" else float(value)) for key, value in payload.items()
    }
    return replace(base, name=f"{base_name}+overrides", **overrides)


def numeric_format_for(compression: Mapping[str, Any] | None) -> str:
    """Infer the deployed numeric format from a compression descriptor.

    Quantisation changes per-cycle MAC throughput by roughly 4x on a scalar
    FPU, so guessing wrong biases the energy estimate by the same factor.
    Anything mentioning quantisation counts as int8; everything else — no
    compression, or pruning alone, which removes weights without changing the
    arithmetic width — stays float32.
    """
    method = str((compression or {}).get("method", "none")).lower()
    return INT8 if "quant" in method else FLOAT32


def estimate_energy(
    *,
    macs: int | None,
    hop_size_s: float = DEFAULT_HOP_SIZE_S,
    profile: EnergyProfile | str | Mapping[str, Any] | None = None,
    numeric_format: str = FLOAT32,
) -> dict[str, Any]:
    """Estimate per-inference energy, average power, and battery life.

    Args:
        macs: Multiply-accumulate count for one inference. ``None`` (torchinfo
            tracing failed) yields a result with null metrics but an intact
            assumption block, so the report still records what was assumed.
        hop_size_s: Seconds between consecutive inferences. Comes from the
            dataset's hop size, since one prediction is emitted per hop.
        profile: Profile, profile name, or override mapping. See
            ``resolve_profile``.
        numeric_format: ``"int8"`` or ``"float32"``. Selects which per-cycle
            throughput figure applies.

    Returns:
        Dict with ``energy_per_inference_mj``, ``avg_power_mw``,
        ``est_battery_life_h``, ``est_battery_life_days``,
        ``active_time_per_inference_ms``, ``duty_cycle``,
        ``real_time_feasible``, ``hop_size_s``, ``numeric_format``, and an
        ``assumptions`` block echoing the full profile.

        ``real_time_feasible`` is False when a single inference takes longer
        than the hop — the device cannot keep up, so ``avg_power_mw`` saturates
        at ``p_active_mw`` and battery life is a floor rather than an estimate.
    """
    if hop_size_s <= 0:
        raise ValueError(f"hop_size_s must be positive, got {hop_size_s}")
    resolved = resolve_profile(profile)
    macs_per_cycle = resolved.macs_per_cycle(numeric_format)
    if macs_per_cycle <= 0:
        raise ValueError(
            f"profile {resolved.name!r} has non-positive macs_per_cycle for {numeric_format!r}"
        )

    assumptions = {
        **resolved.to_dict(),
        "macs_per_cycle_used": macs_per_cycle,
        "model": (
            "E = P_active * t_active + P_sleep * t_sleep, "
            "t_active = MACs / (macs_per_cycle * f_clock). "
            "Compute only: excludes sensor sampling, radio, and memory-movement "
            "energy. Proportional to MACs by construction, so it does not rank "
            "models differently from the gmacs field."
        ),
    }
    if macs is None:
        return {
            "energy_per_inference_mj": None,
            "avg_power_mw": None,
            "est_battery_life_h": None,
            "est_battery_life_days": None,
            "active_time_per_inference_ms": None,
            "duty_cycle": None,
            "real_time_feasible": None,
            "hop_size_s": float(hop_size_s),
            "numeric_format": numeric_format,
            "assumptions": assumptions,
        }

    t_active_s = float(macs) / (macs_per_cycle * resolved.f_clock_hz)
    # mW * s == mJ, so no unit scaling is needed here.
    energy_per_inference_mj = resolved.p_active_mw * t_active_s
    duty_cycle = t_active_s / hop_size_s
    feasible = duty_cycle <= 1.0
    if feasible:
        avg_power_mw = (
            resolved.p_active_mw * duty_cycle + resolved.p_sleep_mw * (1.0 - duty_cycle)
        )
    else:
        # Saturated: the core never sleeps and still falls behind.
        avg_power_mw = resolved.p_active_mw
    # mAh * V == mWh, and mWh / mW == h.
    battery_life_h = (
        resolved.battery_capacity_mah * resolved.battery_voltage_v / avg_power_mw
        if avg_power_mw > 0
        else None
    )

    return {
        "energy_per_inference_mj": round(energy_per_inference_mj, 6),
        "avg_power_mw": round(avg_power_mw, 6),
        "est_battery_life_h": None if battery_life_h is None else round(battery_life_h, 3),
        "est_battery_life_days": None if battery_life_h is None else round(battery_life_h / 24.0, 3),
        "active_time_per_inference_ms": round(t_active_s * 1000.0, 6),
        "duty_cycle": round(duty_cycle, 6),
        "real_time_feasible": feasible,
        "hop_size_s": float(hop_size_s),
        "numeric_format": numeric_format,
        "assumptions": assumptions,
    }
