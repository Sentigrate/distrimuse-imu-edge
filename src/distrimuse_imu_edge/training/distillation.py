from __future__ import annotations

from pathlib import Path

import torch

from distrimuse_imu_edge.training.runner import load_checkpoint_model


def load_teacher(path: str | Path, *, device: torch.device) -> torch.nn.Module:
    model, _ = load_checkpoint_model(path, map_location=device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
