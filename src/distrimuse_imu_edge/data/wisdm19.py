from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WISDM19_URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)

WISDM_ACTIVITY_CODE_TO_NAME: dict[str, str] = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "brushing_teeth",
    "H": "eating_soup",
    "I": "eating_chips",
    "J": "eating_pasta",
    "K": "drinking_from_cup",
    "L": "eating_sandwich",
    "M": "kicking_soccer_ball",
    "O": "playing_catch",
    "P": "dribbling_basketball",
    "Q": "writing",
    "R": "clapping",
    "S": "folding_clothes",
}
WISDM_ACTIVITY_CODES = tuple(WISDM_ACTIVITY_CODE_TO_NAME)
WISDM_ACTIVITY_CODE_TO_ID = {code: idx for idx, code in enumerate(WISDM_ACTIVITY_CODES)}

DEFAULT_TRAIN_SUBJECTS = list(range(1600, 1640))
DEFAULT_VAL_SUBJECTS = list(range(1640, 1645))
DEFAULT_TEST_SUBJECTS = list(range(1645, 1651))


@dataclass(frozen=True, slots=True)
class WISDMSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    @property
    def all_subjects(self) -> list[int]:
        return sorted(set(self.train + self.val + self.test))


DEFAULT_SPLIT = WISDMSplit(
    train=DEFAULT_TRAIN_SUBJECTS,
    val=DEFAULT_VAL_SUBJECTS,
    test=DEFAULT_TEST_SUBJECTS,
)


def _timestamp_to_seconds(values: pd.Series) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(out)
    if finite.any():
        unique = np.unique(np.sort(out[finite]))
        diffs = np.diff(unique)
        diffs = diffs[diffs > 0]
        if len(diffs):
            median_step = float(np.median(diffs))
            if median_step > 1e6:
                out[finite] /= 1e9
            elif median_step > 1e3:
                out[finite] /= 1e6
            elif median_step > 1:
                out[finite] /= 1e3
        else:
            scale = float(np.nanmax(np.abs(out[finite])))
            while scale > 2e10:
                out[finite] /= 1000.0
                scale = float(np.nanmax(np.abs(out[finite])))
    return out


def read_wisdm_raw_file(path: str | Path, *, axis_prefix: str) -> pd.DataFrame:
    """Read one WISDM raw sensor file.

    WISDM raw rows are comma-separated with a trailing semicolon on the z-axis:
    ``subject,activity_code,timestamp,x,y,z;``.
    """
    path = Path(path)
    rows: list[tuple[int, str, int, float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.rstrip(";").split(",")
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_no}: expected 6 comma-separated fields")
            subject, code, timestamp, x, y, z = [part.strip() for part in parts]
            if code not in WISDM_ACTIVITY_CODE_TO_ID:
                raise ValueError(f"{path}:{line_no}: unknown WISDM activity code {code!r}")
            rows.append((int(subject), code, int(timestamp), float(x), float(y), float(z)))

    df = pd.DataFrame(
        rows,
        columns=[
            "person_id",
            "activity_code",
            "timestamp_raw",
            f"{axis_prefix}_x",
            f"{axis_prefix}_y",
            f"{axis_prefix}_z",
        ],
    )
    if df.empty:
        return df
    df["timestamp_s"] = _timestamp_to_seconds(df["timestamp_raw"])
    df["wisdm_activity_id"] = df["activity_code"].map(WISDM_ACTIVITY_CODE_TO_ID).astype(np.int64)
    df["wisdm_activity"] = df["activity_code"].map(WISDM_ACTIVITY_CODE_TO_NAME)
    return df


def align_watch_accel_gyro(
    accel: pd.DataFrame,
    gyro: pd.DataFrame,
    *,
    tolerance_s: float = 0.05,
) -> pd.DataFrame:
    """Align watch accelerometer and gyroscope samples by subject/activity/time."""
    if accel.empty or gyro.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "scenario_id",
                "time",
                "acc_x",
                "acc_y",
                "acc_z",
                "gyr_x",
                "gyr_y",
                "gyr_z",
                "wisdm_activity_id",
                "wisdm_activity",
            ]
        )

    frames: list[pd.DataFrame] = []
    group_cols = ["person_id", "activity_code"]
    for (person_id, activity_code), acc_group in accel.groupby(group_cols, sort=True):
        gyr_group = gyro[
            (gyro["person_id"] == person_id) & (gyro["activity_code"] == activity_code)
        ]
        if gyr_group.empty:
            continue
        acc_group = acc_group.sort_values("timestamp_s").reset_index(drop=True)
        gyr_group = gyr_group.sort_values("timestamp_s").reset_index(drop=True)
        merged = pd.merge_asof(
            acc_group,
            gyr_group[
                [
                    "timestamp_s",
                    "gyr_x",
                    "gyr_y",
                    "gyr_z",
                ]
            ],
            on="timestamp_s",
            direction="nearest",
            tolerance=tolerance_s,
        ).dropna(subset=["gyr_x", "gyr_y", "gyr_z"])
        if merged.empty:
            continue
        t0 = float(merged["timestamp_s"].min())
        merged["time"] = merged["timestamp_s"] - t0
        merged["scenario_id"] = int(WISDM_ACTIVITY_CODE_TO_ID[str(activity_code)])
        frames.append(
            merged[
                [
                    "person_id",
                    "scenario_id",
                    "time",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "gyr_x",
                    "gyr_y",
                    "gyr_z",
                    "wisdm_activity_id",
                    "wisdm_activity",
                ]
            ]
        )

    if not frames:
        raise ValueError(
            "No aligned WISDM watch ACC/GYR samples found. "
            "Increase --align-tolerance-s or verify the raw files."
        )
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = ["time", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    out[numeric_cols] = out[numeric_cols].astype(np.float32)
    out["person_id"] = out["person_id"].astype(np.int64)
    out["scenario_id"] = out["scenario_id"].astype(np.int64)
    out["wisdm_activity_id"] = out["wisdm_activity_id"].astype(np.int64)
    return out.sort_values(["person_id", "scenario_id", "time"]).reset_index(drop=True)


def validate_subject_split(split: WISDMSplit = DEFAULT_SPLIT) -> None:
    groups = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_name, right_name in pairs:
        overlap = groups[left_name] & groups[right_name]
        if overlap:
            raise ValueError(
                f"WISDM split overlap between {left_name} and {right_name}: {sorted(overlap)}"
            )


def write_subject_splits(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    split: WISDMSplit = DEFAULT_SPLIT,
) -> dict[str, Path]:
    validate_subject_split(split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, subjects in {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }.items():
        part = frame[frame["person_id"].isin(subjects)].copy()
        path = output_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths[name] = path

    metadata = {
        "source": "UCI WISDM Smartphone and Smartwatch Activity and Biometrics Dataset",
        "source_url": WISDM19_URL,
        "license": "CC BY 4.0",
        "sensor_subset": "watch accelerometer + watch gyroscope",
        "activity_code_to_id": WISDM_ACTIVITY_CODE_TO_ID,
        "activity_code_to_name": WISDM_ACTIVITY_CODE_TO_NAME,
        "split_subjects": {
            "train": split.train,
            "val": split.val,
            "test": split.test,
        },
        "rows": {
            "train": int(frame["person_id"].isin(split.train).sum()),
            "val": int(frame["person_id"].isin(split.val).sum()),
            "test": int(frame["person_id"].isin(split.test).sum()),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def _download_zip(zip_path: Path, *, url: str = WISDM19_URL, force: bool = False) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not force:
        return zip_path
    urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        marker.write_text(zip_path.name, encoding="utf-8")

    for nested_zip in sorted(extract_dir.rglob("*.zip")):
        nested_marker = nested_zip.with_name(f".{nested_zip.name}.extracted")
        if nested_marker.exists():
            continue
        with zipfile.ZipFile(nested_zip) as zf:
            zf.extractall(nested_zip.parent)
        nested_marker.write_text(nested_zip.name, encoding="utf-8")
    return extract_dir


def _find_sensor_files(extract_dir: Path, *, sensor: str) -> list[Path]:
    if sensor not in {"accel", "gyro"}:
        raise ValueError(f"unsupported WISDM sensor {sensor!r}")
    candidates = [
        p
        for p in extract_dir.rglob("*.txt")
        if "watch" in p.name.lower()
        and sensor in p.name.lower()
        and "data_" in p.name.lower()
    ]
    return sorted(candidates)


def load_wisdm_watch_frame(extract_dir: str | Path, *, tolerance_s: float = 0.05) -> pd.DataFrame:
    extract_dir = Path(extract_dir)
    accel_files = _find_sensor_files(extract_dir, sensor="accel")
    gyro_files = _find_sensor_files(extract_dir, sensor="gyro")
    if not accel_files:
        raise FileNotFoundError(f"no WISDM watch accelerometer files found under {extract_dir}")
    if not gyro_files:
        raise FileNotFoundError(f"no WISDM watch gyroscope files found under {extract_dir}")

    accel = pd.concat(
        [read_wisdm_raw_file(path, axis_prefix="acc") for path in accel_files],
        ignore_index=True,
    )
    gyro = pd.concat(
        [read_wisdm_raw_file(path, axis_prefix="gyr") for path in gyro_files],
        ignore_index=True,
    )
    return align_watch_accel_gyro(accel, gyro, tolerance_s=tolerance_s)


def prepare_wisdm19(
    *,
    root: str | Path = "cache/public/wisdm19",
    source_zip: str | Path | None = None,
    extracted_dir: str | Path | None = None,
    force_download: bool = False,
    tolerance_s: float = 0.05,
) -> dict[str, Path]:
    root = Path(root)
    zip_path = Path(source_zip) if source_zip is not None else root / "raw" / "wisdm-dataset.zip"
    extract_path = Path(extracted_dir) if extracted_dir is not None else root / "extracted"
    if extracted_dir is None:
        if source_zip is None:
            zip_path = _download_zip(zip_path, force=force_download)
        _extract_zip(zip_path, extract_path)

    frame = load_wisdm_watch_frame(extract_path, tolerance_s=tolerance_s)
    return write_subject_splits(frame, root / "splits" / "watch_accel_gyro")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare WISDM-19 watch ACC+GYR splits.")
    parser.add_argument("--root", default="cache/public/wisdm19")
    parser.add_argument("--source-zip", default=None, help="Existing WISDM ZIP to use instead of downloading.")
    parser.add_argument("--extracted-dir", default=None, help="Existing extracted WISDM directory.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--align-tolerance-s", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = prepare_wisdm19(
        root=args.root,
        source_zip=args.source_zip,
        extracted_dir=args.extracted_dir,
        force_download=args.force_download,
        tolerance_s=args.align_tolerance_s,
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
