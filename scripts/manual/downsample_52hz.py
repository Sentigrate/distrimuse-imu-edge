"""Decimate a real IMU-edge split from 104 Hz to ~52 Hz.

Takes every other sample within each (person_id, scenario_id) session so
decimation never straddles a session boundary, then recomputes the
per-row magnitude columns (exact, not approximated, since they only
depend on that same row's acc/gyr values).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SRC_DIR = Path("../distrimuse-early-fusion/cache/datasets/v35/imu")
OUT_DIR = Path("../distrimuse-early-fusion/cache/datasets/v35_52hz/imu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    df = pd.read_parquet(SRC_DIR / f"{split}.parquet")
    print(f"{split}: {len(df)} rows before decimation")

    decimated = df.groupby(["person_id", "scenario_id"], group_keys=False).apply(
        lambda g: g.iloc[::2]
    )

    decimated["acc_magn"] = np.sqrt(
        decimated["acc_x"] ** 2 + decimated["acc_y"] ** 2 + decimated["acc_z"] ** 2
    )
    decimated["gyr_magn"] = np.sqrt(
        decimated["gyr_x"] ** 2 + decimated["gyr_y"] ** 2 + decimated["gyr_z"] ** 2
    )

    decimated = decimated.sort_values(["person_id", "scenario_id", "time"]).reset_index(drop=True)
    print(f"{split}: {len(decimated)} rows after decimation (~52 Hz)")

    decimated.to_parquet(OUT_DIR / f"{split}.parquet", index=False)
    print(f"Saved {OUT_DIR / f'{split}.parquet'}")
