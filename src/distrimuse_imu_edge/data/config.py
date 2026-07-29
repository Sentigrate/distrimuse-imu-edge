from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRAIN_IDS = [2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25]
DEFAULT_VAL_IDS = [1, 3, 6]
DEFAULT_TEST_IDS = [8, 15, 24, 26, 27]
DEFAULT_SCENARIO_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_SENSOR_COLS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
DEFAULT_WINDOW_SIZE_S = 3.0
DEFAULT_HOP_SIZE_S = 1.0


@dataclass(slots=True)
class SplitConfig:
    train_ids: list[int] = field(default_factory=lambda: list(DEFAULT_TRAIN_IDS))
    val_ids: list[int] = field(default_factory=lambda: list(DEFAULT_VAL_IDS))
    test_ids: list[int] = field(default_factory=lambda: list(DEFAULT_TEST_IDS))
    scenario_ids: list[int] = field(default_factory=lambda: list(DEFAULT_SCENARIO_IDS))

    @property
    def person_ids(self) -> list[int]:
        return sorted(set(self.train_ids + self.val_ids + self.test_ids))


@dataclass(slots=True)
class DataConfig:
    campaign: str = "dc-extern-2026-01"
    processing_version: str | None = None
    split_dir: Path | None = None
    manifest_path: Path | None = None
    cache_dir: Path = Path("cache")
    window_cache_dir: Path = Path("cache/windows")
    window_size_s: float = DEFAULT_WINDOW_SIZE_S
    hop_size_s: float = DEFAULT_HOP_SIZE_S
    context_len: int = 8
    future_context_len: int = 0
    task_col: str = "big_movement"
    n_classes: int = 9
    sensor_cols: tuple[str, ...] = DEFAULT_SENSOR_COLS
    batch_size: int = 128
    num_workers: int = 4
    reuse_window_cache: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)

    @property
    def total_context_len(self) -> int:
        return self.context_len + self.future_context_len

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("split_dir", "manifest_path", "cache_dir", "window_cache_dir"):
            out[key] = None if out[key] is None else str(out[key])
        return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _as_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _split_from_payload(payload: dict[str, Any]) -> SplitConfig:
    split_section = payload.get("split", {})
    scenarios_section = payload.get("scenarios", {})
    return SplitConfig(
        train_ids=list(split_section.get("train", DEFAULT_TRAIN_IDS)),
        val_ids=list(split_section.get("val", DEFAULT_VAL_IDS)),
        test_ids=list(split_section.get("test", DEFAULT_TEST_IDS)),
        scenario_ids=list(scenarios_section.get("default", DEFAULT_SCENARIO_IDS)),
    )


def data_config_from_mapping(payload: dict[str, Any], *, base_dir: Path | None = None) -> DataConfig:
    data = dict(payload.get("data", payload))
    if base_dir is None:
        base_dir = Path.cwd()
    split_cfg = data.pop("split", None)
    if isinstance(split_cfg, SplitConfig):
        split = split_cfg
    elif isinstance(split_cfg, dict):
        split = SplitConfig(
            train_ids=list(split_cfg.get("train_ids", split_cfg.get("train", DEFAULT_TRAIN_IDS))),
            val_ids=list(split_cfg.get("val_ids", split_cfg.get("val", DEFAULT_VAL_IDS))),
            test_ids=list(split_cfg.get("test_ids", split_cfg.get("test", DEFAULT_TEST_IDS))),
            scenario_ids=list(split_cfg.get("scenario_ids", split_cfg.get("scenarios", DEFAULT_SCENARIO_IDS))),
        )
    else:
        split = SplitConfig()

    return DataConfig(
        campaign=str(data.get("campaign", "dc-extern-2026-01")),
        processing_version=data.get("processing_version"),
        split_dir=_as_path(data.get("split_dir")),
        manifest_path=_as_path(data.get("manifest_path")),
        cache_dir=_as_path(data.get("cache_dir")) or Path("cache"),
        window_cache_dir=_as_path(data.get("window_cache_dir")) or Path("cache/windows"),
        window_size_s=float(data.get("window_size_s", DEFAULT_WINDOW_SIZE_S)),
        hop_size_s=float(data.get("hop_size_s", DEFAULT_HOP_SIZE_S)),
        context_len=int(data.get("context_len", 8)),
        future_context_len=int(data.get("future_context_len", 0)),
        task_col=str(data.get("task_col", "big_movement")),
        n_classes=int(data.get("n_classes", 9)),
        sensor_cols=tuple(data.get("sensor_cols", DEFAULT_SENSOR_COLS)),
        batch_size=int(data.get("batch_size", 128)),
        num_workers=int(data.get("num_workers", 4)),
        reuse_window_cache=bool(data.get("reuse_window_cache", True)),
        split=split,
    )


def load_config(path: str | Path) -> tuple[DataConfig, dict[str, Any]]:
    """Load a YAML config and return ``(DataConfig, full_payload)``.

    The optional ``defaults.split`` key points to a split YAML, resolved relative
    to the config file. Values in the main config win over defaults.
    """
    cfg_path = Path(path).expanduser().resolve()
    payload = _read_yaml(cfg_path)
    defaults = payload.get("defaults", {}) or {}
    split_payload: dict[str, Any] = {}
    if "split" in defaults:
        split_path = Path(defaults["split"])
        if not split_path.is_absolute():
            split_path = cfg_path.parent.parent / split_path if str(split_path).startswith("configs/") else cfg_path.parent / split_path
        split_payload = _read_yaml(split_path)
    payload.setdefault("data", {})
    if split_payload:
        split = asdict(_split_from_payload(split_payload))
        explicit_split = payload["data"].get("split", {}) or {}
        if isinstance(explicit_split, dict):
            split.update(explicit_split)
        payload["data"]["split"] = split
    return data_config_from_mapping(payload, base_dir=cfg_path.parent), payload
