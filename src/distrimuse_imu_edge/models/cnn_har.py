from __future__ import annotations

import torch
from torch import nn

from distrimuse_imu_edge.models.base import CurrentWindowMixin
from distrimuse_imu_edge.models.registry import register_model


@register_model("cnn_har")
class CNNHAR(nn.Module, CurrentWindowMixin):
    """CNN-HAR style temporal Conv2D classifier for a single IMU window."""

    def __init__(
        self,
        n_classes: int = 9,
        input_channels: int = 6,
        width_mult: float | None = None,
    ) -> None:
        super().__init__()
        _ = width_mult
        self.input_channels = int(input_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1)),
            nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0),
            nn.Conv2d(50, 40, kernel_size=(5, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1)),
            nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0),
            nn.Conv2d(40, 20, kernel_size=(3, 1)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0),
            nn.Conv2d(20, 400, kernel_size=(1, self.input_channels)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.classifier = nn.Conv2d(400, n_classes, kernel_size=1)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        window = self.current_window(x)
        if window.shape[-1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got window shape {tuple(window.shape)}"
            )
        if window.shape[1] < 22:
            raise ValueError("cnn_har requires at least 22 time steps per window")
        return window.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self._to_image(x))
        return self.classifier(x).squeeze(-1).squeeze(-1)
