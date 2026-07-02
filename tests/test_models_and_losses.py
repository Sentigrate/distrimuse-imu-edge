from __future__ import annotations

import torch

from distrimuse_imu_edge.models import build_model, list_models
from distrimuse_imu_edge.training.losses import distillation_loss


def test_registered_models_emit_class_logits() -> None:
    assert {"teacher_causal_cnn", "edge_cnn", "edge_tcn", "cnn_har", "tinierhar"}.issubset(set(list_models()))
    x = torch.randn(2, 8, 20, 6)
    for name in ("edge_cnn", "edge_tcn"):
        model = build_model(name, n_classes=9, input_channels=6, width_mult=0.25)
        assert model(x).shape == (2, 9)

    teacher = build_model(
        "teacher_causal_cnn",
        n_classes=9,
        input_channels=6,
        context_len=8,
        width_mult=0.25,
        d_model=48,
        nhead=3,
        num_layers=1,
        enc_channels=32,
    )
    assert teacher(x).shape == (2, 9)


def test_whar_reference_style_models_emit_class_logits() -> None:
    x = torch.randn(2, 8, 64, 6)
    for name in ("cnn_har", "tinierhar"):
        model = build_model(name, n_classes=9, input_channels=6, width_mult=0.25)
        assert model(x).shape == (2, 9)


def test_distillation_loss_combines_ce_and_kl() -> None:
    student = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.5, -0.5]], requires_grad=True)
    teacher = torch.tensor([[3.0, 0.1, -2.0], [0.1, 2.0, -1.0]])
    target = torch.tensor([0, 1])

    ce_only = distillation_loss(student, target)
    distilled = distillation_loss(student, target, teacher_logits=teacher, temperature=4.0, alpha=0.5)

    assert ce_only.item() > 0
    assert distilled.item() > 0
    assert distilled.item() != ce_only.item()
