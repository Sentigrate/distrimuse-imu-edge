from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from distrimuse_imu_edge.data.config import DataConfig, load_config
from distrimuse_imu_edge.training.config import TrainConfig, train_config_from_mapping

SINGLE_WINDOW_MODELS = frozenset({"cnn_har", "tinierhar", "edge_cnn", "edge_tcn"})
WINDOW_SEQUENCE_MODELS = frozenset(
    {
        "teacher_causal_cnn",
        "causal_context_transformer_cnn",
        "edge_window_gru",
        "edge_window_tcn",
    }
)


def load_runtime_config(config_path: str | Path) -> tuple[DataConfig, TrainConfig, dict[str, Any]]:
    data_cfg, payload = load_config(config_path)
    train_cfg = train_config_from_mapping(payload)
    resolved = {
        "data": data_cfg.to_dict(),
        "train": train_cfg.to_dict(),
        "raw": payload,
    }
    return data_cfg, train_cfg, resolved


def effective_context_len_for(model_name: str, configured_context_len: int) -> int:
    if model_name in SINGLE_WINDOW_MODELS:
        return 1
    return configured_context_len


def effective_context_lengths_for(
    model_name: str,
    configured_context_len: int,
    configured_future_context_len: int,
) -> tuple[int, int]:
    if model_name in SINGLE_WINDOW_MODELS:
        return 1, 0
    return configured_context_len, configured_future_context_len


def model_kwargs_for(model_name: str, *, data_cfg: DataConfig, width_mult: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n_classes": data_cfg.n_classes,
        "input_channels": len(data_cfg.sensor_cols),
        "width_mult": width_mult,
    }
    if model_name in WINDOW_SEQUENCE_MODELS:
        kwargs["current_index"] = data_cfg.context_len - 1
        kwargs["bidirectional"] = data_cfg.future_context_len > 0
    if model_name in {"teacher_causal_cnn", "causal_context_transformer_cnn"}:
        kwargs["context_len"] = data_cfg.total_context_len
    return kwargs


def default_run_name(
    model_name: str,
    *,
    width_mult: float,
    context_len: int,
    future_context_len: int = 0,
    suffix: str | None = None,
) -> str:
    base = f"{model_name}_wm{width_mult:g}_ctx{context_len}"
    if future_context_len:
        base = f"{base}_future{future_context_len}"
    return f"{base}_{suffix}" if suffix else base


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
