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
            BIG_MOVEMENT_CLASS_NAMES[index] if index < len(BIG_MOVEMENT_CLASS_NAMES) else str(index)
            for index in range(n_classes)
        ]
    return [str(index) for index in range(n_classes)]


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader, *, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
        )
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


from distrimuse_core.metrics_events import (
    EventExtractionConfig,
    IoUMatchConfig,
    extract_events,
    match_events,
)


def event_classification_report_payload(
    predictions: pd.DataFrame,
    *,
    n_classes: int = 9,
    prefix: str = "test",
    extraction_cfg: EventExtractionConfig | None = None,
    match_cfg: IoUMatchConfig | None = None,
) -> dict:
    """Per-class event-level Precision/Recall/F1, dataset-level aggregate.

    Mirrors classification_report_payload's flat-dict shape, but scores
    contiguous same-class runs of windows ("events") matched by temporal
    IoU instead of individual windows. See
    distrimuse_core.metrics_events for the extraction/matching logic this
    wraps.

    Events are extracted per (person_id, scenario_id) group, not on the
    whole frame at once: extract_events assumes one continuous timeline,
    and window_start_s can overlap across subjects/scenarios, so grouping
    first avoids merging or matching events across unrelated recordings.
    tp/fp/fn are accumulated across all groups before computing precision/
    recall/F1, giving the correct dataset-level metric rather than an
    average of per-subject F1 values.

    Args:
        predictions: The same frame written to predictions.parquet. Must
            contain y_true, y_pred, person_id, scenario_id, window_start_s,
            and (if it holds more than one split) a split column.
        n_classes: Number of classes (rows in the per-class output).
        prefix: Split name, used both to filter predictions["split"] (if
            present) and to prefix output keys, matching
            classification_report_payload's convention.
        extraction_cfg: Passed to extract_events. Defaults to
            EventExtractionConfig() (1s min duration, 1s merge gap).
        match_cfg: Passed to match_events. Defaults to IoUMatchConfig()
            (IoU >= 0.3).

    Returns:
        Dict with f"{prefix}_event_macro_f1", f"{prefix}_event_per_class_f1",
        f"{prefix}_event_per_class_precision", f"{prefix}_event_per_class_recall",
        f"{prefix}_event_true_counts", f"{prefix}_event_pred_counts" -- same
        flat-dict shape as classification_report_payload so both merge into
        the same metrics.json.
    """
    extraction_cfg = extraction_cfg or EventExtractionConfig()
    match_cfg = match_cfg or IoUMatchConfig()
    classes = list(range(n_classes))
    class_names = class_names_for(n_classes)

    df = predictions
    if "split" in df.columns:
        df = df[df["split"] == prefix]

    agg_tp = {c: 0 for c in classes}
    agg_fp = {c: 0 for c in classes}
    agg_fn = {c: 0 for c in classes}
    agg_true_count = {c: 0 for c in classes}
    agg_pred_count = {c: 0 for c in classes}

    for _, group in df.groupby(["person_id", "scenario_id"]):
        group = group.sort_values("window_start_s")
        timestamps = group["window_start_s"].to_numpy()
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()

        true_events = extract_events(y_true, timestamps, extraction_cfg)
        pred_events = extract_events(y_pred, timestamps, extraction_cfg)
        result = match_events(true_events, pred_events, match_cfg)

        for cid in classes:
            true_c = [e for e in true_events if e.class_id == cid]
            pred_c = [e for e in pred_events if e.class_id == cid]
            matched_c = [(t, p) for t, p in result.matched if t.class_id == cid]
            tp = len(matched_c)
            agg_tp[cid] += tp
            agg_fp[cid] += len(pred_c) - tp
            agg_fn[cid] += len(true_c) - tp
            agg_true_count[cid] += len(true_c)
            agg_pred_count[cid] += len(pred_c)

    per_class_f1: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for cid in classes:
        tp, fp, fn = agg_tp[cid], agg_fp[cid], agg_fn[cid]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        name = class_names[cid] if cid < len(class_names) else str(cid)
        per_class_f1[name] = round(f1, 4)
        per_class_precision[name] = round(prec, 4)
        per_class_recall[name] = round(rec, 4)

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    name_of = lambda c: class_names[c] if c < len(class_names) else str(c)  # noqa: E731

    return {
        f"{prefix}_event_macro_f1": round(macro_f1, 4),
        f"{prefix}_event_per_class_f1": per_class_f1,
        f"{prefix}_event_per_class_precision": per_class_precision,
        f"{prefix}_event_per_class_recall": per_class_recall,
        f"{prefix}_event_true_counts": {name_of(c): agg_true_count[c] for c in classes},
        f"{prefix}_event_pred_counts": {name_of(c): agg_pred_count[c] for c in classes},
    }
