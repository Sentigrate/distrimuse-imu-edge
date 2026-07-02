from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from distrimuse_imu_edge.data.datamodule import IMUEdgeDataModule
from distrimuse_imu_edge.training.config import TrainConfig
from distrimuse_imu_edge.training.runner import train_model


def finetune_compressed_model(
    *,
    model: torch.nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    datamodule: IMUEdgeDataModule,
    train_config: TrainConfig,
    output_dir: Path,
    resolved_config: dict[str, Any],
    compression: dict[str, Any],
) -> dict[str, Any]:
    return train_model(
        model=model,
        model_name=model_name,
        model_kwargs=model_kwargs,
        datamodule=datamodule,
        train_config=train_config,
        output_dir=output_dir,
        resolved_config=resolved_config,
        compression=compression,
    )
