from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from distrimuse_imu_edge.data.config import DataConfig, SplitConfig, load_config
from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.data.wisdm19 import (
    WISDM_ACTIVITY_CODE_TO_ID,
    WISDMSplit,
    _extract_zip,
    align_watch_accel_gyro,
    read_wisdm_raw_file,
    validate_subject_split,
    write_subject_splits,
)
from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.training.config import TrainConfig
from distrimuse_imu_edge.training.runner import load_transfer_checkpoint, train_model


def _write_raw(path: Path, *, subject: int, code: str, offset: int = 0) -> None:
    lines = []
    for idx in range(12):
        timestamp = 1_000_000_000 + offset + idx * 50_000_000
        lines.append(f"{subject},{code},{timestamp},{idx}.0,{idx + 1}.0,{idx + 2}.0;")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_wisdm_raw_parser_and_alignment(tmp_path) -> None:
    acc_path = tmp_path / "data_1600_accel_watch.txt"
    gyr_path = tmp_path / "data_1600_gyro_watch.txt"
    _write_raw(acc_path, subject=1600, code="A", offset=0)
    _write_raw(gyr_path, subject=1600, code="A", offset=10_000_000)

    acc = read_wisdm_raw_file(acc_path, axis_prefix="acc")
    gyr = read_wisdm_raw_file(gyr_path, axis_prefix="gyr")
    aligned = align_watch_accel_gyro(acc, gyr, tolerance_s=0.02)

    assert aligned.columns.tolist() == [
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
    assert set(aligned["person_id"]) == {1600}
    assert set(aligned["wisdm_activity_id"]) == {WISDM_ACTIVITY_CODE_TO_ID["A"]}
    assert set(aligned["wisdm_activity"]) == {"walking"}
    assert aligned["time"].iloc[0] == 0.0
    assert len(aligned) == 12


def test_extract_zip_unpacks_nested_archives_after_outer_marker_exists(tmp_path) -> None:
    inner_zip = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip, "w") as zf:
        zf.writestr("raw/watch/accel/data_1600_accel_watch.txt", "payload")

    outer_zip = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.write(inner_zip, arcname="wisdm-dataset.zip")

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    (extract_dir / ".extracted").write_text("outer.zip", encoding="utf-8")
    (extract_dir / "wisdm-dataset.zip").write_bytes(inner_zip.read_bytes())

    _extract_zip(outer_zip, extract_dir)

    assert (extract_dir / "raw" / "watch" / "accel" / "data_1600_accel_watch.txt").exists()


def test_wisdm_subject_split_validation_and_writes(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "person_id": [1600, 1601, 1602],
            "scenario_id": [0, 0, 0],
            "time": [0.0, 0.0, 0.0],
            "acc_x": [0.0, 0.0, 0.0],
            "acc_y": [0.0, 0.0, 0.0],
            "acc_z": [0.0, 0.0, 0.0],
            "gyr_x": [0.0, 0.0, 0.0],
            "gyr_y": [0.0, 0.0, 0.0],
            "gyr_z": [0.0, 0.0, 0.0],
            "wisdm_activity_id": [0, 0, 0],
            "wisdm_activity": ["walking", "walking", "walking"],
        }
    )
    split = WISDMSplit(train=[1600], val=[1601], test=[1602])

    validate_subject_split(split)
    paths = write_subject_splits(frame, tmp_path / "splits", split=split)

    assert set(paths) == {"train", "val", "test"}
    assert pd.read_parquet(paths["train"])["person_id"].tolist() == [1600]
    metadata = json.loads((tmp_path / "splits" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["split_subjects"]["test"] == [1602]


def test_wisdm_subject_split_rejects_overlap() -> None:
    try:
        validate_subject_split(WISDMSplit(train=[1600], val=[1600], test=[1601]))
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping WISDM split was not rejected")


def test_pretrain_wisdm_config_keeps_public_subject_split() -> None:
    cfg, _ = load_config("configs/pretrain_wisdm19.yaml")

    assert cfg.split.train_ids[0] == 1600
    assert cfg.split.train_ids[-1] == 1639
    assert cfg.split.val_ids == [1640, 1641, 1642, 1643, 1644]
    assert cfg.split.test_ids[-1] == 1650
    assert cfg.split.scenario_ids == list(range(18))


def test_transfer_checkpoint_loads_body_and_skips_head(tmp_path) -> None:
    source = build_model("edge_cnn", n_classes=18, input_channels=6, width_mult=0.25)
    with torch.no_grad():
        source.features[0].weight.fill_(0.25)
    ckpt_path = tmp_path / "wisdm.ckpt"
    torch.save(
        {
            "model_name": "edge_cnn",
            "model_kwargs": {"n_classes": 18, "input_channels": 6, "width_mult": 0.25},
            "state_dict": source.state_dict(),
        },
        ckpt_path,
    )

    target = build_model("edge_cnn", n_classes=9, input_channels=6, width_mult=0.25)
    report = load_transfer_checkpoint(target, ckpt_path)

    assert "features.0.weight" in report["loaded_keys"]
    assert "head.2.weight" in report["skipped_head"]
    assert torch.allclose(target.features[0].weight, torch.full_like(target.features[0].weight, 0.25))
    assert target.head[2].out_features == 9


def _participant_frame(person_id: int, label: int, *, task_col: str, n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(person_id + label)
    frame = pd.DataFrame(
        {
            "person_id": person_id,
            "scenario_id": label,
            "time": np.arange(n_rows) / 100.0,
            task_col: label,
        }
    )
    for col in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
        frame[col] = rng.normal(size=n_rows).astype(np.float32)
    return frame


def _write_split_parquets(split_dir: Path, *, task_col: str, labels: tuple[int, int, int]) -> None:
    split_dir.mkdir(parents=True)
    for split_name, person_id, label in zip(
        ("train", "val", "test"),
        (1600, 1640, 1645),
        labels,
        strict=True,
    ):
        _participant_frame(person_id, label, task_col=task_col).to_parquet(
            split_dir / f"{split_name}.parquet",
            index=False,
        )


def test_wisdm_pretrain_then_distrimuse_finetune_smoke(tmp_path) -> None:
    wisdm_split_dir = tmp_path / "wisdm"
    _write_split_parquets(wisdm_split_dir, task_col="wisdm_activity_id", labels=(0, 1, 1))
    wisdm_cfg = DataConfig(
        split_dir=wisdm_split_dir,
        window_cache_dir=tmp_path / "wisdm-window-cache",
        window_size_s=0.2,
        hop_size_s=0.1,
        context_len=3,
        task_col="wisdm_activity_id",
        n_classes=18,
        batch_size=4,
        num_workers=0,
        reuse_window_cache=False,
        split=SplitConfig(train_ids=[1600], val_ids=[1640], test_ids=[1645], scenario_ids=[0, 1]),
    )
    wisdm_dm = IMUEdgeDataModule(wisdm_cfg)
    wisdm_dm.setup()
    pretrain_kwargs = {"n_classes": 18, "input_channels": 6, "width_mult": 0.25}
    pretrain_out = tmp_path / "pretrain"
    train_model(
        model=build_model("edge_cnn", **pretrain_kwargs),
        model_name="edge_cnn",
        model_kwargs=pretrain_kwargs,
        datamodule=wisdm_dm,
        train_config=TrainConfig(max_epochs=1, early_stop_patience=1, output_root=tmp_path, device="cpu"),
        output_dir=pretrain_out,
        resolved_config={"data": wisdm_cfg.to_dict()},
    )

    distrimuse_split_dir = tmp_path / "distrimuse"
    _write_split_parquets(distrimuse_split_dir, task_col="big_movement", labels=(0, 1, 1))
    distrimuse_cfg = DataConfig(
        split_dir=distrimuse_split_dir,
        window_cache_dir=tmp_path / "distrimuse-window-cache",
        window_size_s=0.2,
        hop_size_s=0.1,
        context_len=3,
        task_col="big_movement",
        n_classes=9,
        batch_size=4,
        num_workers=0,
        reuse_window_cache=False,
        split=SplitConfig(train_ids=[1600], val_ids=[1640], test_ids=[1645], scenario_ids=[0, 1]),
    )
    distrimuse_dm = IMUEdgeDataModule(distrimuse_cfg)
    distrimuse_dm.setup()
    finetune_kwargs = {"n_classes": 9, "input_channels": 6, "width_mult": 0.25}
    finetune_model = build_model("edge_cnn", **finetune_kwargs)
    transfer = load_transfer_checkpoint(finetune_model, pretrain_out / "checkpoints" / "best.ckpt")

    assert transfer["loaded_count"] > 0
    assert transfer["skipped_head"]

    finetune_out = tmp_path / "finetune"
    train_model(
        model=finetune_model,
        model_name="edge_cnn",
        model_kwargs=finetune_kwargs,
        datamodule=distrimuse_dm,
        train_config=TrainConfig(max_epochs=1, early_stop_patience=1, output_root=tmp_path, device="cpu"),
        output_dir=finetune_out,
        resolved_config={"data": distrimuse_cfg.to_dict(), "init_checkpoint": transfer},
    )

    assert (finetune_out / "checkpoints" / "best.ckpt").exists()
    assert (finetune_out / "reports" / "metrics.json").exists()
