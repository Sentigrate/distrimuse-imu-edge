from __future__ import annotations

import torch
from torch import nn


class ContextSlicingStudent(nn.Module):
    """Run a narrow-context student on batches shaped for a wide-context teacher.

    Privileged-context distillation feeds the dataloader's full
    ``[B, N_teacher, T, C]`` sequence to the teacher while the student only reads
    ``length`` windows starting at ``start``. Slicing here instead of inside the
    training loop keeps every evaluation, report, and profiling path working on
    the batch the dataloader already produces.

    A batch that already contains exactly ``length`` windows is passed straight
    through, so efficiency profiling can measure the deployed student input shape.

    ``state_dict`` and ``load_state_dict`` delegate to the wrapped student, so a
    saved checkpoint loads back into a bare student with no adapter present.
    """

    def __init__(self, student: nn.Module, *, start: int, length: int) -> None:
        super().__init__()
        if start < 0:
            raise ValueError("start must be >= 0")
        if length < 1:
            raise ValueError("length must be >= 1")
        self.student = student
        self.start = int(start)
        self.length = int(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B,N,T,C), got {tuple(x.shape)}")
        windows = x.shape[1]
        if windows == self.length:
            return self.student(x)
        if windows < self.start + self.length:
            raise ValueError(
                f"sequence has {windows} windows; student reads indices "
                f"{self.start}..{self.start + self.length - 1}"
            )
        return self.student(x[:, self.start : self.start + self.length])

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[override]
        return self.student.load_state_dict(state_dict, *args, **kwargs)
