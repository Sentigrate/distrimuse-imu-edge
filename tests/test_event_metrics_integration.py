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


def test_training_writes_event_based_metrics(tmp_path) -> None:
    """metrics.json should carry event-level F1 alongside window-level F1.

    Mirrors test_smoke_training.py's report-contract check, but for the
    event_classification_report_payload fields wired into train_model:
    macro F1, per-class F1/precision/recall, and true/pred event counts,
    for both the val and test splits.
    """
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

    metrics = json.loads((out / "reports" / "metrics.json").read_text(encoding="utf-8"))

    for split in ("val", "test"):
        assert f"{split}_event_macro_f1" in metrics
        assert isinstance(metrics[f"{split}_event_macro_f1"], float)

        per_class_f1 = metrics[f"{split}_event_per_class_f1"]
        assert len(per_class_f1) == data_cfg.n_classes
        assert all(0.0 <= v <= 1.0 for v in per_class_f1.values())

        per_class_precision = metrics[f"{split}_event_per_class_precision"]
        per_class_recall = metrics[f"{split}_event_per_class_recall"]
        assert set(per_class_precision) == set(per_class_f1)
        assert set(per_class_recall) == set(per_class_f1)

        true_counts = metrics[f"{split}_event_true_counts"]
        pred_counts = metrics[f"{split}_event_pred_counts"]
        assert set(true_counts) == set(per_class_f1)
        assert all(isinstance(v, int) for v in true_counts.values())
        assert all(isinstance(v, int) for v in pred_counts.values())
