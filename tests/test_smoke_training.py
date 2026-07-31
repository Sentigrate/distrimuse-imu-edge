from __future__ import annotations

import json

import numpy as np
import pandas as pd

from distrimuse_imu_edge.data.config import DataConfig, SplitConfig
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.config import TrainConfig
from distrimuse_imu_edge.training.runner import train_model


def _participant_frame(person_id: int, label: int) -> pd.DataFrame:
    rows = 80
    rng = np.random.default_rng(person_id)
    frame = pd.DataFrame(
        {
            "person_id": person_id,
            "scenario_id": 1,
            "time": np.arange(rows) / 100.0,
            "big_movement": label,
        }
    )
    for col in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
        frame[col] = rng.normal(size=rows).astype(np.float32)
    return frame


def test_one_epoch_training_writes_report_contract(tmp_path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _participant_frame(1, 0).to_parquet(split_dir / "train.parquet", index=False)
    _participant_frame(2, 1).to_parquet(split_dir / "val.parquet", index=False)
    _participant_frame(3, 1).to_parquet(split_dir / "test.parquet", index=False)

    data_cfg = DataConfig(
        split_dir=split_dir,
        window_cache_dir=tmp_path / "window-cache",
        window_size_s=0.2,
        hop_size_s=0.1,
        context_len=3,
        n_classes=3,
        batch_size=4,
        num_workers=0,
        reuse_window_cache=False,
        split=SplitConfig(train_ids=[1], val_ids=[2], test_ids=[3], scenario_ids=[1]),
    )
    dm = IMUEdgeDataModule(data_cfg)
    dm.setup()
    model_kwargs = {
        "n_classes": 3,
        "input_channels": 6,
        "width_mult": 0.25,
        "current_index": 2,
        "bidirectional": False,
    }
    model = build_model("edge_window_tcn", **model_kwargs)
    train_cfg = TrainConfig(max_epochs=1, early_stop_patience=1, output_root=tmp_path, device="cpu")
    out = tmp_path / "run"

    train_model(
        model=model,
        model_name="edge_window_tcn",
        model_kwargs=model_kwargs,
        datamodule=dm,
        train_config=train_cfg,
        output_dir=out,
        resolved_config={"data": data_cfg.to_dict(), "train": train_cfg.to_dict()},
    )

    assert (out / "checkpoints" / "best.ckpt").exists()
    assert (out / "reports" / "metrics.json").exists()
    assert (out / "reports" / "model_stats.json").exists()
    assert (out / "reports" / "predictions.parquet").exists()
    assert (out / "reports" / "config.resolved.yaml").exists()
    metrics = json.loads((out / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert "test_macro_f1" in metrics
    # resolved_config above carries no "energy" key, so this also pins the
    # default-profile fallback for callers that predate energy reporting.
    stats = json.loads((out / "reports" / "model_stats.json").read_text(encoding="utf-8"))
    assert stats["energy"]["energy_per_inference_mj"] > 0
    assert stats["energy"]["assumptions"]["name"] == "nrf52840_m4f_64mhz"
    assert stats["energy"]["hop_size_s"] == data_cfg.hop_size_s
