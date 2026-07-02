from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from distrimuse_core.constants import FS as DEFAULT_FS
except Exception:  # pragma: no cover - import fallback for isolated unit tests
    DEFAULT_FS = 104


class ChannelNormalizer:
    """Per-channel z-score normalizer fitted on raw IMU windows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ChannelNormalizer":
        if x.size == 0:
            raise ValueError("cannot fit normalizer on an empty array")
        self.mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.std = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() must be called before transform()")
        return ((x - self.mean[None, None, :]) / self.std[None, None, :]).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        if self.mean is None or self.std is None:
            return {}
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, state: dict[str, Iterable[float]]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)


def stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _resample_window(window: np.ndarray, target_len: int) -> np.ndarray:
    if len(window) == target_len:
        return window.astype(np.float32, copy=False)
    if len(window) < 2:
        return np.repeat(window.astype(np.float32), target_len, axis=0)
    src = np.linspace(0.0, 1.0, len(window), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    out = np.empty((target_len, window.shape[1]), dtype=np.float32)
    for col in range(window.shape[1]):
        out[:, col] = np.interp(dst, src, window[:, col].astype(np.float64)).astype(np.float32)
    return out


def _to_seconds(values: pd.Series | np.ndarray) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or series.dtype == object:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        if dt.notna().sum() >= 2:
            out = np.full(len(series), np.nan, dtype=np.float64)
            valid = dt.notna().to_numpy()
            out[valid] = dt.astype("int64").to_numpy(dtype=np.float64)[valid] / 1e9
            finite = np.isfinite(out)
            out[finite] -= float(np.nanmin(out[finite]))
            return out
    out = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(out)
    if finite.any():
        scale = float(np.nanmax(np.abs(out[finite])))
        while scale > 2e10:
            out[finite] /= 1000.0
            scale = float(np.nanmax(np.abs(out[finite])))
        out[finite] -= float(np.nanmin(out[finite]))
    return out


def _absolute_timestamps_s(df: pd.DataFrame) -> np.ndarray | None:
    for col in ("timestamp_dt", "timestamp", "Timestamp"):
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], utc=True, errors="coerce")
        if dt.notna().sum() < 2:
            continue
        out = np.full(len(df), np.nan, dtype=np.float64)
        valid = dt.notna().to_numpy()
        out[valid] = dt.astype("int64").to_numpy(dtype=np.float64)[valid] / 1e9
        return out
    return None


def _time_seconds(df: pd.DataFrame, fs: float) -> np.ndarray:
    for col in ("imu_s", "timestamp_dt", "timestamp", "Timestamp", "timestamp_s", "time", "Time"):
        if col in df.columns:
            t = _to_seconds(df[col])
            finite = np.isfinite(t)
            if finite.sum() >= 2 and float(np.nanmax(t[finite]) - np.nanmin(t[finite])) > 0:
                return t
    return np.arange(len(df), dtype=np.float64) / fs


def _continuous_slices(t: np.ndarray, *, window_size_s: float) -> list[slice]:
    if len(t) == 0:
        return []
    max_gap_s = max(5.0, 2.0 * window_size_s)
    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(t)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def _recording_groups(df: pd.DataFrame):
    if "scenario_id" not in df.columns:
        yield -1, df
        return
    for scenario_id, scenario_df in df.groupby("scenario_id", sort=False, dropna=False):
        yield int(scenario_id) if pd.notna(scenario_id) else -1, scenario_df


def _build_manifest_windows(
    *,
    recording_df: pd.DataFrame,
    t: np.ndarray,
    t_abs: np.ndarray | None,
    sensor_values: np.ndarray,
    person_id: int,
    scenario_id: int,
    window_manifest: pd.DataFrame,
    expected_len: int,
    windows: list[np.ndarray],
    labels: list[int],
    starts: list[float],
) -> None:
    if t_abs is None or not np.isfinite(t_abs).any():
        return
    session_manifest = window_manifest[
        (window_manifest["person_id"] == person_id)
        & (window_manifest["scenario_id"] == scenario_id)
    ].sort_values("window_start_ns")
    if session_manifest.empty:
        return
    for row in session_manifest.itertuples(index=False):
        start_s = int(row.window_start_ns) * 1e-9
        end_s = int(row.window_end_ns) * 1e-9
        i0 = int(np.searchsorted(t_abs, start_s, side="left"))
        i1 = int(np.searchsorted(t_abs, end_s, side="left"))
        if i1 - i0 < 3:
            continue
        windows.append(_resample_window(sensor_values[i0:i1], expected_len))
        labels.append(int(row.big_movement))
        starts.append(start_s)


def build_raw_window_dataset(
    df_split: pd.DataFrame,
    participants: list[int],
    *,
    task_col: str,
    window_size_s: float,
    hop_size_s: float,
    sensor_cols: list[str] | tuple[str, ...],
    fs: float = DEFAULT_FS,
    majority_threshold: float = 0.6,
    window_manifest: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build raw IMU windows and preserve participant/scenario grouping."""
    expected_len = int(round(window_size_s * fs))
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    s_all: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []

    for person_id in participants:
        person_df = df_split[df_split["person_id"] == person_id].copy()
        if person_df.empty:
            continue
        person_windows: list[np.ndarray] = []
        person_labels: list[int] = []
        person_scenarios: list[int] = []
        person_starts: list[float] = []

        for scenario_id, recording_df in _recording_groups(person_df):
            before = len(person_labels)
            recording_df = recording_df.copy().reset_index(drop=True)
            t = _time_seconds(recording_df, fs)
            t_abs = _absolute_timestamps_s(recording_df)
            valid = np.isfinite(t)
            if valid.sum() < 3:
                continue
            if not valid.all():
                recording_df = recording_df.loc[valid].reset_index(drop=True)
                t = t[valid]
                if t_abs is not None:
                    t_abs = t_abs[valid]
            order = np.argsort(t, kind="stable")
            if not np.array_equal(order, np.arange(len(recording_df))):
                recording_df = recording_df.iloc[order].reset_index(drop=True)
                t = t[order]
                if t_abs is not None:
                    t_abs = t_abs[order]

            sensor_values = recording_df[list(sensor_cols)].to_numpy(dtype=np.float32)
            if window_manifest is not None:
                _build_manifest_windows(
                    recording_df=recording_df,
                    t=t,
                    t_abs=t_abs,
                    sensor_values=sensor_values,
                    person_id=person_id,
                    scenario_id=scenario_id,
                    window_manifest=window_manifest,
                    expected_len=expected_len,
                    windows=person_windows,
                    labels=person_labels,
                    starts=person_starts,
                )
                person_scenarios.extend([scenario_id] * (len(person_labels) - before))
                continue

            if task_col not in recording_df.columns:
                raise KeyError(f"task column not found: {task_col}")
            labels = recording_df[task_col].to_numpy(dtype=np.int64)
            for segment in _continuous_slices(t, window_size_s=window_size_s):
                segment_t = t[segment]
                segment_labels = labels[segment]
                segment_sensor = sensor_values[segment]
                segment_abs = t_abs[segment] if t_abs is not None else None
                if len(segment_t) < 3:
                    continue
                start_s = float(segment_t[0])
                end_limit = float(segment_t[-1])
                while start_s + window_size_s <= end_limit:
                    end_s = start_s + window_size_s
                    i0 = int(np.searchsorted(segment_t, start_s, side="left"))
                    i1 = int(np.searchsorted(segment_t, end_s, side="left"))
                    if i1 - i0 < 3:
                        start_s += hop_size_s
                        continue
                    y_window = segment_labels[i0:i1]
                    if np.any(y_window < 0):
                        start_s += hop_size_s
                        continue
                    uniq, counts = np.unique(y_window, return_counts=True)
                    maj_idx = int(np.argmax(counts))
                    if counts[maj_idx] / counts.sum() >= majority_threshold:
                        person_windows.append(_resample_window(segment_sensor[i0:i1], expected_len))
                        person_labels.append(int(uniq[maj_idx]))
                        if segment_abs is not None and i0 < len(segment_abs) and np.isfinite(segment_abs[i0]):
                            person_starts.append(float(segment_abs[i0]))
                        else:
                            person_starts.append(float(start_s))
                    start_s += hop_size_s
            person_scenarios.extend([scenario_id] * (len(person_labels) - before))

        if person_labels:
            n = len(person_labels)
            x_all.append(np.stack(person_windows).astype(np.float32))
            y_all.append(np.asarray(person_labels, dtype=np.int64))
            p_all.append(np.full(n, person_id, dtype=np.int64))
            s_all.append(np.asarray(person_scenarios, dtype=np.int64))
            ts_all.append(np.asarray(person_starts, dtype=np.float64))

    channels = len(sensor_cols)
    if not x_all:
        return (
            np.empty((0, expected_len, channels), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.concatenate(x_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(p_all, axis=0),
        np.concatenate(s_all, axis=0),
        np.concatenate(ts_all, axis=0),
    )


def group_counts(person_ids: np.ndarray, scenario_ids: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    if len(person_ids) == 0:
        return pd.DataFrame(columns=["person_id", "scenario_id", "n_windows"])
    start = 0
    current = (int(person_ids[0]), int(scenario_ids[0]))
    for idx, pair in enumerate(zip(person_ids.tolist(), scenario_ids.tolist(), strict=True)):
        pair_i = (int(pair[0]), int(pair[1]))
        if pair_i != current:
            rows.append({"person_id": current[0], "scenario_id": current[1], "n_windows": idx - start})
            start = idx
            current = pair_i
    rows.append({"person_id": current[0], "scenario_id": current[1], "n_windows": len(person_ids) - start})
    return pd.DataFrame(rows)


def save_window_cache(path: Path, *, x: np.ndarray, y: np.ndarray, pids: np.ndarray, sids: np.ndarray, starts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=x, y=y, person_ids=pids, scenario_ids=sids, window_starts_s=starts)


def load_window_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["X"], data["y"], data["person_ids"], data["scenario_ids"], data["window_starts_s"]
