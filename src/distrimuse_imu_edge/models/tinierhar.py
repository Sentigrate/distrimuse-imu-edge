from __future__ import annotations

import torch
from torch import nn

from distrimuse_imu_edge.models.base import CurrentWindowMixin
from distrimuse_imu_edge.models.registry import register_model


class _DepthwiseSeparableTemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _TinierHARBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pool: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _DepthwiseSeparableTemporalConv(in_channels, out_channels, kernel_size=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        shortcut_layers: list[nn.Module] = []
        if in_channels != out_channels:
            shortcut_layers.extend([nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)])
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 1)))
            shortcut_layers.append(nn.MaxPool2d(kernel_size=(2, 1)))

        self.block = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(*shortcut_layers) if shortcut_layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


@register_model("tinierhar")
class TinierHAR(nn.Module, CurrentWindowMixin):
    """TinierHAR-style depthwise Conv2D, BiGRU, and attention classifier."""

    def __init__(
        self,
        n_classes: int = 9,
        input_channels: int = 6,
        width_mult: float | None = None,
        nb_filters: int = 4,
        nb_conv_blocks: int = 4,
        nb_units_gru: int = 16,
        drop_prob: float = 0.3,
    ) -> None:
        super().__init__()
        _ = width_mult
        self.input_channels = int(input_channels)
        blocks: list[nn.Module] = [
            _TinierHARBlock(1, nb_filters, pool=True),
            _TinierHARBlock(nb_filters, 2 * nb_filters, pool=True),
        ]
        blocks.extend(_TinierHARBlock(2 * nb_filters, 2 * nb_filters, pool=False) for _ in range(nb_conv_blocks))
        self.conv_blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(drop_prob)
        self.gru = nn.GRU(
            input_size=2 * nb_filters * self.input_channels,
            hidden_size=nb_units_gru,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.Linear(2 * nb_units_gru, 1)
        self.classifier = nn.Linear(2 * nb_units_gru, n_classes)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        window = self.current_window(x)
        if window.shape[-1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got window shape {tuple(window.shape)}"
            )
        return window.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(self._to_image(x))
        batch_size, channels, time_steps, sensor_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * sensor_channels)
        x = self.dropout(x)
        x, _ = self.gru(x)
        weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(weights * x, dim=1)
        return self.classifier(x)
