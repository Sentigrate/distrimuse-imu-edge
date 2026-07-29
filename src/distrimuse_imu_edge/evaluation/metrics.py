from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

try:
    from distrimuse_core.labels import BIG_MOVEMENT_CLASS_NAMES
except Exception:  # pragma: no cover
    BIG_MOVEMENT_CLASS_NAMES = [str(i) for i in range(9)]


def class_names_for(n_classes: int, *, task_col: str = "big_movement") -> list[str]:
    if task_col == "big_movement":
        return [
            BIG_MOVEMENT_CLASS_NAMES[index]
            if index < len(BIG_MOVEMENT_CLASS_NAMES)
            else str(index)
            for index in range(n_classes)
        ]
    return [str(index) for index in range(n_classes)]


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)
        y_true.append(y.numpy())
        y_pred.append(prob.argmax(dim=-1).cpu().numpy())
        y_prob.append(prob.cpu().numpy())
    if not y_true:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def classification_report_payload(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_classes: int = 9,
    prefix: str = "test",
) -> dict:
    labels = list(range(n_classes))
    class_names = class_names_for(n_classes)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    payload: dict = {
        f"{prefix}_macro_f1": float(macro),
        f"{prefix}_per_class_f1": {
            class_names[i] if i < len(class_names) else str(i): float(per_class[i])
            for i in range(n_classes)
        },
        f"{prefix}_confusion_matrix": cm.astype(int).tolist(),
    }
    if y_prob is not None and y_prob.size:
        payload[f"{prefix}_mean_confidence"] = float(np.max(y_prob, axis=1).mean())
    return payload


def predictions_frame(
    *,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split,
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "person_id": metadata["person_ids"].astype(int),
            "scenario_id": metadata["scenario_ids"].astype(int),
            "window_start_s": metadata["window_starts_s"].astype(float),
        }
    )
    for idx in range(y_prob.shape[1] if y_prob.ndim == 2 else 0):
        frame[f"prob_{idx}"] = y_prob[:, idx]
    return frame
