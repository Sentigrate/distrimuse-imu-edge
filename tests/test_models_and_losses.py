from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from distrimuse_imu_edge.models import build_model, list_models
from distrimuse_imu_edge.models.edge_window_sequence import _SequenceConv
from distrimuse_imu_edge.training.losses import distillation_loss


def test_registered_models_emit_class_logits() -> None:
    assert {
        "teacher_causal_cnn",
        "edge_cnn",
        "edge_tcn",
        "edge_window_gru",
        "edge_window_tcn",
        "cnn_har",
        "tinierhar",
    }.issubset(set(list_models()))
    x = torch.randn(2, 8, 20, 6)
    for name in ("edge_window_gru", "edge_window_tcn"):
        model = build_model(
            name,
            n_classes=9,
            input_channels=6,
            width_mult=0.25,
            current_index=7,
        )
        assert model(x).shape == (2, 9)

    current = x[:, -1:]
    for name in ("edge_cnn", "edge_tcn"):
        model = build_model(name, n_classes=9, input_channels=6, width_mult=0.25)
        assert model(current).shape == (2, 9)
        with pytest.raises(ValueError, match="single-window model"):
            model(x)

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


def test_teacher_selects_current_and_future_attention_is_explicit() -> None:
    x = torch.randn(2, 5, 20, 6)
    changed_future = x.clone()
    changed_future[:, 3:] += 10.0
    kwargs = {
        "n_classes": 3,
        "input_channels": 6,
        "context_len": 5,
        "current_index": 2,
        "width_mult": 0.25,
        "d_model": 48,
        "nhead": 3,
        "num_layers": 1,
        "enc_channels": 32,
        "dropout": 0.0,
    }

    causal = build_model("teacher_causal_cnn", **kwargs, bidirectional=False).eval()
    with torch.no_grad():
        causal_before = causal(x)
        causal_after = causal(changed_future)
    assert torch.allclose(causal_before, causal_after, atol=1e-6)

    bidirectional = build_model("teacher_causal_cnn", **kwargs, bidirectional=True).eval()
    with torch.no_grad():
        bidirectional_before = bidirectional(x)
        bidirectional_after = bidirectional(changed_future)
    assert not torch.allclose(bidirectional_before, bidirectional_after)


@pytest.mark.parametrize("model_name", ["edge_window_gru", "edge_window_tcn"])
def test_causal_window_students_ignore_future_tokens(model_name: str) -> None:
    model = build_model(
        model_name,
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=2,
        bidirectional=False,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, 20, 6)
    changed_future = x.clone()
    changed_future[:, 3:] += 10.0

    with torch.no_grad():
        before = model(x)
        after = model(changed_future)

    assert torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("model_name", ["edge_window_gru", "edge_window_tcn"])
def test_bidirectional_window_students_use_future_tokens(model_name: str) -> None:
    model = build_model(
        model_name,
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=2,
        bidirectional=True,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, 20, 6)
    changed_future = x.clone()
    changed_future[:, 3:] += 10.0

    with torch.no_grad():
        before = model(x)
        after = model(changed_future)

    assert not torch.allclose(before, after)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_single_position_sequence_convolution_matches_conv1d(bidirectional: bool) -> None:
    layer = _SequenceConv(4, 5, dilation=2, bidirectional=bidirectional)
    x = torch.randn(3, 4, 1, requires_grad=True)

    actual = layer(x)
    expected = functional.conv1d(
        x if bidirectional else functional.pad(x, (4, 0)),
        layer.conv.weight,
        layer.conv.bias,
        padding=2 if bidirectional else 0,
        dilation=2,
    )

    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert layer.conv.weight.grad is not None


def test_single_window_tcn_supports_forward_and_backward() -> None:
    model = build_model(
        "edge_window_tcn",
        n_classes=3,
        input_channels=6,
        width_mult=0.25,
        current_index=0,
    )
    output = model(torch.randn(2, 1, 20, 6))

    assert output.shape == (2, 3)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_whar_reference_style_models_emit_class_logits() -> None:
    x = torch.randn(2, 1, 64, 6)
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
