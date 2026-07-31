from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score

from distrimuse_core.metrics import (
    TimelineTask,
    make_cm_figure,
    make_prediction_timeline_figure,
)


def _safe(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        )
    )


def _normalised_cm(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)


def _write_cm(
    path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    class_names: list[str],
) -> None:
    figure = make_cm_figure(y_true, title, class_names, y_pred=y_pred)
    figure.write_html(path, full_html=True, include_plotlyjs="cdn")


def _subject_cm_overview_figure(
    *,
    test: pd.DataFrame,
    person_ids: list[int],
    class_names: list[str],
) -> go.Figure:
    n_classes = len(class_names)
    panels: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "All test subjects",
            test["y_true"].to_numpy(dtype=int),
            test["y_pred"].to_numpy(dtype=int),
        )
    ]
    for person_id in person_ids:
        subject = test[test["person_id"] == person_id]
        panels.append(
            (
                f"Subject {person_id}",
                subject["y_true"].to_numpy(dtype=int),
                subject["y_pred"].to_numpy(dtype=int),
            )
        )

    n_cols = min(3, len(panels))
    n_rows = math.ceil(len(panels) / n_cols)
    titles = [
        f"{name}<br>macro-F1={_macro_f1(y_true, y_pred, n_classes=n_classes):.3f}"
        for name, y_true, y_pred in panels
    ]
    figure = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.07,
        vertical_spacing=min(0.18, 0.38 / n_rows),
    )

    for index, (_, y_true, y_pred) in enumerate(panels):
        row = (index // n_cols) + 1
        col = (index % n_cols) + 1
        is_bottom_row = row == n_rows
        is_left_column = col == 1
        cm = _normalised_cm(y_true, y_pred, n_classes=n_classes)
        figure.add_trace(
            go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=index == 0,
                text=[[f"{value:.2f}" for value in values] for values in cm],
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate=(
                    "True: %{y}<br>Predicted: %{x}<br>Row-normalised: %{z:.3f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        figure.update_xaxes(
            title_text="Predicted" if is_bottom_row else "",
            tickangle=-35,
            tickfont_size=8,
            showticklabels=is_bottom_row,
            automargin=True,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text="True" if is_left_column else "",
            autorange="reversed",
            tickfont_size=8,
            showticklabels=is_left_column,
            automargin=True,
            row=row,
            col=col,
        )

    for annotation in figure.layout.annotations:
        annotation.font.size = 14

    figure.update_layout(
        title={
            "text": "Test confusion matrices by subject (row-normalised)",
            "x": 0.5,
            "xanchor": "center",
        },
        height=max(620, 510 * n_rows),
        width=max(720, 500 * n_cols),
        margin={"l": 90, "r": 90, "t": 120, "b": 110},
    )
    return figure


def _write_subject_cm_overview(
    path: Path,
    *,
    test: pd.DataFrame,
    person_ids: list[int],
    class_names: list[str],
) -> None:
    figure = _subject_cm_overview_figure(
        test=test,
        person_ids=person_ids,
        class_names=class_names,
    )
    figure.write_html(path, full_html=True, include_plotlyjs="cdn")


def _timeline_frame(subject: pd.DataFrame, *, hop_size_s: float) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = subject.sort_values(
        ["scenario_id", "window_start_s"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    timeline_parts: list[np.ndarray] = []
    offset = 0.0
    for _, scenario in ordered.groupby("scenario_id", sort=False, dropna=False):
        starts = scenario["window_start_s"].to_numpy(dtype=float)
        if len(starts) and np.isfinite(starts).all():
            relative = starts - starts[0]
        else:
            relative = np.arange(len(scenario), dtype=float) * hop_size_s
        timeline_parts.append(offset + relative)
        offset += (float(relative[-1]) if len(relative) else 0.0) + hop_size_s
    return ordered, np.concatenate(timeline_parts)


def write_test_prediction_artifacts(
    *,
    output_dir: Path,
    predictions: pd.DataFrame,
    class_names: list[str],
    hop_size_s: float,
) -> None:
    required = {"split", "y_true", "y_pred", "person_id", "scenario_id", "window_start_s"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")

    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return

    n_classes = len(class_names)
    person_ids = sorted(int(value) for value in test["person_id"].unique())
    cms_dir = output_dir / "confusion_matrices"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    cms_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_true = test["y_true"].to_numpy(dtype=int)
    all_pred = test["y_pred"].to_numpy(dtype=int)
    all_f1 = _macro_f1(all_true, all_pred, n_classes=n_classes)
    _write_cm(
        cms_dir / "test_all_subjects.html",
        y_true=all_true,
        y_pred=all_pred,
        title=f"All test subjects — macro-F1={all_f1:.3f}",
        class_names=class_names,
    )

    subject_metrics: list[dict[str, int | float]] = []
    for person_id in person_ids:
        subject = test[test["person_id"] == person_id]
        y_true = subject["y_true"].to_numpy(dtype=int)
        y_pred = subject["y_pred"].to_numpy(dtype=int)
        macro_f1 = _macro_f1(y_true, y_pred, n_classes=n_classes)
        subject_metrics.append(
            {
                "person_id": person_id,
                "n_windows": len(subject),
                "test_macro_f1": macro_f1,
            }
        )
        _write_cm(
            cms_dir / f"test_subject_{_safe(person_id)}.html",
            y_true=y_true,
            y_pred=y_pred,
            title=f"Test subject {person_id} — macro-F1={macro_f1:.3f}",
            class_names=class_names,
        )

    _write_subject_cm_overview(
        cms_dir / "test_subjects_overview.html",
        test=test,
        person_ids=person_ids,
        class_names=class_names,
    )
    pd.DataFrame(subject_metrics).to_csv(reports_dir / "test_per_subject_metrics.csv", index=False)
    (reports_dir / "test_per_subject_metrics.json").write_text(
        json.dumps(subject_metrics, indent=2),
        encoding="utf-8",
    )

    probability_columns = sorted(
        (column for column in test.columns if column.startswith("prob_")),
        key=lambda column: int(column.removeprefix("prob_")),
    )
    if len(probability_columns) != n_classes:
        return

    for person_id in person_ids:
        subject = test[test["person_id"] == person_id]
        ordered, times = _timeline_frame(subject, hop_size_s=hop_size_s)
        task = TimelineTask(
            name=f"Subject {person_id}",
            true_labels=ordered["y_true"].to_numpy(dtype=int),
            prob=ordered[probability_columns].to_numpy(dtype=float),
            class_names=class_names,
        )
        figure = make_prediction_timeline_figure(
            times,
            task,
            title=f"Prediction timeline — test subject {person_id}",
        )
        figure.write_html(
            plots_dir / f"prediction_timeline_subject_{_safe(person_id)}.html",
            full_html=True,
            include_plotlyjs="cdn",
        )
