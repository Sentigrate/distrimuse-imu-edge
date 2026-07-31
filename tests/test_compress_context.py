from __future__ import annotations

import pytest

from distrimuse_imu_edge.cli.compress import (
    check_context_matches_checkpoint,
    resolve_context_lengths,
)
from distrimuse_imu_edge.data.config import DataConfig


def _ckpt(context_len: int, future_context_len: int) -> dict:
    return {"config": {"data": {"context_len": context_len, "future_context_len": future_context_len}}}


def test_checkpoint_context_wins_over_config() -> None:
    """The config's context is irrelevant to a checkpoint trained on another one."""
    cfg = DataConfig(context_len=8, future_context_len=0)

    assert resolve_context_lengths(
        _ckpt(1, 0), data_cfg=cfg, context_len=None, future_context_len=None
    ) == (1, 0)
    assert resolve_context_lengths(
        _ckpt(8, 7), data_cfg=cfg, context_len=None, future_context_len=None
    ) == (8, 7)


def test_cli_values_win_over_checkpoint() -> None:
    cfg = DataConfig(context_len=8, future_context_len=0)

    assert resolve_context_lengths(
        _ckpt(1, 0), data_cfg=cfg, context_len=4, future_context_len=2
    ) == (4, 2)


def test_config_used_when_checkpoint_records_no_context() -> None:
    cfg = DataConfig(context_len=8, future_context_len=3)

    assert resolve_context_lengths(
        {}, data_cfg=cfg, context_len=None, future_context_len=None
    ) == (8, 3)


def test_guard_accepts_matching_context() -> None:
    check_context_matches_checkpoint(
        {"current_index": 7, "bidirectional": False}, context_len=8, future_context_len=0
    )
    check_context_matches_checkpoint(
        {"current_index": 7, "bidirectional": True}, context_len=8, future_context_len=7
    )
    check_context_matches_checkpoint(
        {"current_index": 0, "bidirectional": False}, context_len=1, future_context_len=0
    )


def test_guard_rejects_wrong_context_len() -> None:
    # A current-only checkpoint (current_index=0) evaluated with 8 windows.
    with pytest.raises(SystemExit, match="current_index"):
        check_context_matches_checkpoint(
            {"current_index": 0, "bidirectional": False}, context_len=8, future_context_len=0
        )


def test_guard_rejects_bidirectional_without_future_context() -> None:
    # The future-7 checkpoint evaluated with the config's future_context_len=0,
    # which would silently strip half its receptive field.
    with pytest.raises(SystemExit, match="bidirectional"):
        check_context_matches_checkpoint(
            {"current_index": 7, "bidirectional": True}, context_len=8, future_context_len=0
        )


def test_guard_ignores_models_without_context_kwargs() -> None:
    """Single-window models carry neither kwarg and must not be rejected."""
    check_context_matches_checkpoint({"width_mult": 0.25}, context_len=1, future_context_len=0)
