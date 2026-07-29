from __future__ import annotations

import torch
from torch import nn

from distrimuse_imu_edge.models.base import CurrentWindowMixin, make_width
from distrimuse_imu_edge.models.registry import register_model


@register_model("edge_cnn")
class EdgeCNN(nn.Module, CurrentWindowMixin):
    """Compact single-window CNN student."""

    def __init__(self, n_classes: int = 9, input_channels: int = 6, width_mult: float = 0.5) -> None:
        super().__init__()
        c1 = make_width(48, width_mult)
        c2 = make_width(96, width_mult)
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(c2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.current_window(x).transpose(1, 2)
        return self.head(self.features(signal))
