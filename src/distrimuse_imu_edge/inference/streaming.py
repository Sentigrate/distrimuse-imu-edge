"""Streaming, embedding-cached inference for context-window models.

Motivation
----------
EdgeWindowTCN / EdgeWindowGRU (see ``models/edge_window_sequence.py``)
predict the label of one "current" window using a fixed context of
``total_context_len`` windows (past, and future for bidirectional models).
The straightforward way to run this at deployment time re-encodes every
window in that context on every hop — but each raw window is encoded by
``window_encoder`` completely independently of the others (see
``_WindowSequenceMixin.encode_windows``), and consecutive hops share almost
all of their context. Re-encoding is therefore redundant: once a window has
been encoded, its embedding is valid forever and can be cached.

``StreamingWindowPredictor`` keeps a fixed-size buffer of already-computed
embeddings. Each call to :meth:`push` encodes only the newest raw window,
slides it into the buffer, and — once the buffer holds a full context —
runs the (much cheaper) temporal block over the buffer to produce a
prediction. Measured with ``compute_streaming_model_stats`` (see
``evaluation/efficiency.py``) against the real width-0.25 checkpoints, this
drops peak activation memory for ``edge_window_tcn`` from 585 KiB to 39 KiB
(15.0x) for the past 7 + current + future 7 context mode, and holds at a flat
39 KiB regardless of context length — 8x lower than the 312 KiB batched
figure for past 7 + current, and unchanged from the 39 KiB current-only
figure, since one window through the encoder is the whole cost either way.
See ``experiments/results/edge_window_tcn_context_comparison.md``'s
"Streaming, embedding-cached inference" section for the full measured
comparison, including latency and int8.

This changes only *how* inference is computed, not *what* it computes:
fed the same stream of raw windows one at a time, :meth:`push` returns the
same logits (up to floating-point reassociation) as a full batched forward
pass over the equivalent context window. See
``tests/test_streaming_equivalence.py`` for the check against the existing
batched ``forward()``.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import nn


class StreamingWindowPredictor:
    """Incremental, embedding-cached wrapper around a window-sequence model.

    One instance holds the state (embedding buffer) for exactly one stream
    — one sensor, one person. Callers handling several simultaneous streams
    should keep one instance per stream, e.g. a ``dict`` keyed by device id;
    this class deliberately does not try to multiplex streams itself, since
    how streams map to devices/people is a deployment concern this repo
    doesn't own.

    Args:
        model: A trained model exposing ``window_encoder``, ``temporal``,
            ``head``, and ``current_index`` — i.e. anything built on
            ``_WindowSequenceMixin`` (``EdgeWindowTCN``, ``EdgeWindowGRU``).
            Moved to eval mode; not moved across devices, so pass a model
            already on the device you want inference to run on.
        total_context_len: Number of windows the model expects as context
            (``context_len + future_context_len``, matching the training
            config). Not inferable from the model object alone, so it must
            be supplied explicitly.

    Attributes:
        current_index: Copied from ``model.current_index``. Position within
            the buffer that corresponds to the window being predicted for.
            For a causal model this is ``total_context_len - 1`` (predict
            for the newest window, no delay). For a bidirectional model
            with ``future_context_len`` look-ahead windows, this is
            ``total_context_len - 1 - future_context_len``: pushing a new
            window only produces a prediction for a window that is
            ``future_context_len`` hops in the past, since the model needs
            that many windows *after* it as future context. That delay is
            inherent to the bidirectional architecture and is unaffected by
            caching — caching only removes redundant re-encoding, not the
            wait for future windows to arrive.
    """

    def __init__(self, model: nn.Module, *, total_context_len: int) -> None:
        if not hasattr(model, "window_encoder") or not hasattr(model, "temporal"):
            raise TypeError(
                f"{type(model).__name__} does not look like a window-sequence "
                "model (missing window_encoder/temporal). Expected something "
                "built on _WindowSequenceMixin, e.g. EdgeWindowTCN."
            )
        if total_context_len <= 0:
            raise ValueError("total_context_len must be positive")

        self.model = model.eval()
        self.current_index: int = int(model.current_index)
        self.total_context_len = int(total_context_len)
        if not 0 <= self.current_index < self.total_context_len:
            raise ValueError(
                f"model.current_index={self.current_index} is outside "
                f"[0, {self.total_context_len})"
            )
        self._buffer: deque[torch.Tensor] = deque(maxlen=self.total_context_len)

    @property
    def is_warmed_up(self) -> bool:
        """Whether the buffer holds enough windows to produce a prediction."""
        return len(self._buffer) == self.total_context_len

    def reset(self) -> None:
        """Clear the embedding buffer, e.g. when starting a new session."""
        self._buffer.clear()

    @torch.no_grad()
    def push(self, raw_window: torch.Tensor) -> torch.Tensor | None:
        """Feed one new raw window; return logits, or None while warming up.

        Args:
            raw_window: Tensor of shape ``(T, C)`` — one window of raw
                sensor samples (``T`` timesteps, ``C`` channels), matching
                what a single position along the context axis holds in the
                batched ``(B, N, T, C)`` input the model was trained on.

        Returns:
            Logits of shape ``(n_classes,)`` once the buffer is full, or
            ``None`` if fewer than ``total_context_len`` windows have been
            pushed so far (there is no valid prediction yet — this mirrors
            a real deployment, which likewise has nothing to predict until
            enough windows have arrived).
        """
        if raw_window.ndim != 2:
            raise ValueError(
                f"expected raw_window of shape (T, C), got {tuple(raw_window.shape)}"
            )

        x = raw_window.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        embedding = self.model.window_encoder(x).squeeze(0)  # (embedding_dim,)
        self._buffer.append(embedding)

        if not self.is_warmed_up:
            return None

        stacked = torch.stack(list(self._buffer), dim=0)  # (N, embedding_dim)
        stacked = stacked.unsqueeze(0).transpose(1, 2)  # (1, embedding_dim, N)
        temporal_out = self.model.temporal(stacked).transpose(
            1, 2
        )  # (1, N, hidden_dim)
        current = temporal_out[:, self.current_index]  # (1, hidden_dim)
        logits = self.model.head(current)  # (1, n_classes)
        return logits.squeeze(0)
