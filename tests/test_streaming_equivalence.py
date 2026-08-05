from __future__ import annotations

import torch

from distrimuse_imu_edge.inference.streaming import StreamingWindowPredictor
from distrimuse_imu_edge.models import build_model


def _run_equivalence_check(
    *, bidirectional: bool, context_len: int, future_context_len: int
) -> None:
    torch.manual_seed(0)
    n_classes, input_channels, width_mult, window_len = 5, 6, 0.25, 20
    total_context_len = context_len + future_context_len
    current_index = context_len - 1

    model_kwargs = {
        "n_classes": n_classes,
        "input_channels": input_channels,
        "width_mult": width_mult,
        "current_index": current_index,
        "bidirectional": bidirectional,
    }
    model = build_model("edge_window_tcn", **model_kwargs).eval()

    n_session_windows = total_context_len + 6
    session = torch.randn(n_session_windows, window_len, input_channels)

    predictor = StreamingWindowPredictor(model, total_context_len=total_context_len)
    streamed = [predictor.push(session[i]) for i in range(n_session_windows)]

    # A prediction that streams out at step i corresponds to absolute window
    # (i - delay), where delay = total_context_len - 1 - current_index is the
    # number of future windows the model needs to have already seen (0 for a
    # causal model, future_context_len for a bidirectional one).
    #
    # Whether push() returns None depends only on whether the buffer has
    # filled yet (i >= total_context_len - 1) — NOT on target_abs_index,
    # which is a separate question of *which* window the prediction is for
    # once the buffer is full.
    delay = total_context_len - 1 - current_index
    for i, logits in enumerate(streamed):
        if i < total_context_len - 1:
            assert logits is None, f"expected None while warming up at step {i}"
            continue
        target_abs_index = i - delay
        start = target_abs_index - current_index
        end = start + total_context_len
        context = session[start:end].unsqueeze(
            0
        )  # (1, N, T, C), matches forward()'s input shape
        with torch.no_grad():
            expected = model(context).squeeze(0)
        assert torch.allclose(logits, expected, atol=1e-5), (
            f"streaming and batched logits diverge at step {i} "
            f"(bidirectional={bidirectional}, delay={delay})"
        )


def test_streaming_matches_batched_forward_causal() -> None:
    """Causal model (context only, no look-ahead): predictions have no delay."""
    _run_equivalence_check(bidirectional=False, context_len=8, future_context_len=0)


def test_streaming_matches_batched_forward_bidirectional() -> None:
    """Bidirectional model: predictions lag by future_context_len hops."""
    _run_equivalence_check(bidirectional=True, context_len=8, future_context_len=8)


def test_streaming_returns_none_until_buffer_is_full() -> None:
    model_kwargs = {
        "n_classes": 3,
        "input_channels": 6,
        "width_mult": 0.25,
        "current_index": 2,
        "bidirectional": False,
    }
    model = build_model("edge_window_tcn", **model_kwargs).eval()
    predictor = StreamingWindowPredictor(model, total_context_len=3)

    assert predictor.push(torch.randn(20, 6)) is None
    assert predictor.push(torch.randn(20, 6)) is None
    assert predictor.push(torch.randn(20, 6)) is not None
    assert predictor.is_warmed_up
