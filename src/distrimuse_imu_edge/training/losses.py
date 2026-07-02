from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return CE loss or CE + softened KL distillation loss."""
    ce = F.cross_entropy(student_logits, target, weight=class_weights)
    if teacher_logits is None:
        return ce
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    t = float(temperature)
    kl = F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)
    return (1.0 - alpha) * ce + alpha * kl
