from __future__ import annotations

from distrimuse_imu_edge.cli.common import (
    effective_context_lengths_for,
    model_kwargs_for,
)
from distrimuse_imu_edge.data.config import DataConfig


def test_single_window_models_force_one_current_window() -> None:
    for model_name in ("edge_cnn", "edge_tcn", "cnn_har", "tinierhar"):
        assert effective_context_lengths_for(model_name, 8, 7) == (1, 0)


def test_context_model_kwargs_mark_current_and_future_mode() -> None:
    config = DataConfig(context_len=8, future_context_len=7)

    teacher = model_kwargs_for("teacher_causal_cnn", data_cfg=config, width_mult=0.5)
    assert teacher["context_len"] == 15
    assert teacher["current_index"] == 7
    assert teacher["bidirectional"] is True

    for model_name in ("edge_window_gru", "edge_window_tcn"):
        kwargs = model_kwargs_for(model_name, data_cfg=config, width_mult=0.5)
        assert kwargs["current_index"] == 7
        assert kwargs["bidirectional"] is True
