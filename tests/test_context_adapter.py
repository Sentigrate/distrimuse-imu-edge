from __future__ import annotations

import pytest
import torch

from distrimuse_imu_edge.models import build_model
from distrimuse_imu_edge.models.context_adapter import ContextSlicingStudent


def _student(**kwargs):
    return build_model(
        "edge_window_tcn",
        n_classes=9,
        input_channels=6,
        width_mult=0.25,
        current_index=0,
        bidirectional=False,
        **kwargs,
    )


def test_slicing_student_ignores_teacher_context():
    """A current-window student must not react to the teacher's past or future."""
    wrapped = ContextSlicingStudent(_student(), start=7, length=1).eval()
    x = torch.randn(4, 15, 312, 6)
    with torch.no_grad():
        baseline = wrapped(x)
        mutated = x.clone()
        mutated[:, :7] = torch.randn_like(mutated[:, :7])
        mutated[:, 8:] = torch.randn_like(mutated[:, 8:])
        assert torch.allclose(baseline, wrapped(mutated), atol=1e-6)
        # The current window is the only input that changes the output.
        mutated[:, 7] = torch.randn_like(mutated[:, 7])
        assert not torch.allclose(baseline, wrapped(mutated), atol=1e-6)


def test_slicing_student_reads_the_current_window():
    student = _student().eval()
    wrapped = ContextSlicingStudent(student, start=7, length=1).eval()
    x = torch.randn(2, 15, 312, 6)
    with torch.no_grad():
        assert torch.allclose(wrapped(x), student(x[:, 7:8]), atol=1e-6)
        # Already student-shaped input passes through, as during efficiency profiling.
        assert torch.allclose(wrapped(x[:, 7:8]), student(x[:, 7:8]), atol=1e-6)


def test_state_dict_round_trips_through_a_bare_student():
    trained = ContextSlicingStudent(_student(), start=7, length=1)
    fresh = _student()
    fresh.load_state_dict(trained.state_dict())
    x = torch.randn(2, 1, 312, 6)
    trained.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(trained(x), fresh(x), atol=1e-6)


def test_short_sequence_is_rejected():
    wrapped = ContextSlicingStudent(_student(), start=7, length=1)
    with pytest.raises(ValueError, match="reads indices"):
        wrapped(torch.randn(2, 4, 312, 6))
