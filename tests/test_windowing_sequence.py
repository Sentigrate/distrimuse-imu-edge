from __future__ import annotations

import numpy as np
import pandas as pd

from distrimuse_imu_edge.data.sequence import CausalSequenceWindowDataset
from distrimuse_imu_edge.data.windowing import build_raw_window_dataset


def test_windowing_builds_fixed_length_windows() -> None:
    rows = 120
    df = pd.DataFrame(
        {
            "person_id": 1,
            "scenario_id": 1,
            "time": np.arange(rows) / 100.0,
            "big_movement": np.zeros(rows, dtype=int),
            "acc_x": np.linspace(0, 1, rows),
            "acc_y": 0.0,
            "acc_z": 0.0,
            "gyr_x": 0.0,
            "gyr_y": 0.0,
            "gyr_z": 0.0,
        }
    )
    x, y, pids, sids, starts = build_raw_window_dataset(
        df,
        [1],
        task_col="big_movement",
        window_size_s=0.2,
        hop_size_s=0.1,
        sensor_cols=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        fs=100,
    )
    assert x.shape[1:] == (20, 6)
    assert len(x) == len(y) == len(pids) == len(sids) == len(starts)
    assert set(y.tolist()) == {0}


def test_causal_context_does_not_cross_scenario_boundary() -> None:
    x = np.ones((4, 5, 6), dtype=np.float32)
    y = np.arange(4, dtype=np.int64)
    pids = np.array([1, 1, 1, 1])
    sids = np.array([1, 1, 2, 2])
    ds = CausalSequenceWindowDataset(x, y, pids, sids, context_len=3)

    _, current_y, context_y = ds[2]

    assert int(current_y) == 2
    assert context_y.tolist() == [-1, -1, 2]
