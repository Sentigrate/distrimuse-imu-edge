from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """Emit past, current, and optional future windows within group boundaries.

    ``context_len`` includes the current window, so ``context_len=8`` means
    seven past windows plus the current window. ``future_context_len`` adds
    look-ahead windows. The returned label is always the current window label.
    Missing context at either group boundary is zero padded and marked with
    ``-1`` in ``context_labels`` for optional sequence losses.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        person_ids: np.ndarray,
        scenario_ids: np.ndarray,
        *,
        context_len: int,
        future_context_len: int = 0,
        window_starts_s: np.ndarray | None = None,
    ) -> None:
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if future_context_len < 0:
            raise ValueError("future_context_len must be >= 0")
        if not (len(x) == len(y) == len(person_ids) == len(scenario_ids)):
            raise ValueError("x, y, person_ids, and scenario_ids must have equal length")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.person_ids = np.asarray(person_ids, dtype=np.int64)
        self.scenario_ids = np.asarray(scenario_ids, dtype=np.int64)
        self.context_len = int(context_len)
        self.future_context_len = int(future_context_len)
        self.total_context_len = self.context_len + self.future_context_len
        self.window_starts_s = (
            np.asarray(window_starts_s, dtype=np.float64)
            if window_starts_s is not None
            else np.full(len(y), np.nan, dtype=np.float64)
        )
        self._index_map = self._build_index_map()

    def _build_index_map(self) -> list[list[int]]:
        index_map: list[list[int]] = []
        for idx in range(len(self.y)):
            group = (self.person_ids[idx], self.scenario_ids[idx])
            indices: list[int] = []
            cursor = idx - self.context_len + 1
            final_cursor = idx + self.future_context_len
            while cursor <= final_cursor:
                if (
                    cursor < 0
                    or cursor >= len(self.y)
                    or (self.person_ids[cursor], self.scenario_ids[cursor]) != group
                ):
                    indices.append(-1)
                else:
                    indices.append(cursor)
                cursor += 1
            index_map.append(indices)
        return index_map

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self._index_map[idx]
        t, c = self.x.shape[1], self.x.shape[2]
        x_seq = torch.zeros((self.total_context_len, t, c), dtype=torch.float32)
        y_seq = torch.full((self.total_context_len,), -1, dtype=torch.long)
        for pos, source_idx in enumerate(indices):
            if source_idx >= 0:
                x_seq[pos] = self.x[source_idx]
                y_seq[pos] = self.y[source_idx]
        return x_seq, self.y[idx], y_seq

    def metadata(self) -> dict[str, np.ndarray]:
        return {
            "person_ids": self.person_ids.copy(),
            "scenario_ids": self.scenario_ids.copy(),
            "window_starts_s": self.window_starts_s.copy(),
        }


# Backwards-compatible import for callers that only use causal context.
CausalSequenceWindowDataset = SequenceWindowDataset
