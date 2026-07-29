from __future__ import annotations

import json

import pandas as pd

from distrimuse_imu_edge.evaluation.artifacts import write_test_prediction_artifacts


def test_test_artifacts_include_combined_and_per_subject_outputs(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "y_true": [0, 1, 2, 0, 1, 2],
            "y_pred": [0, 1, 1, 0, 2, 2],
            "person_id": [8, 8, 8, 15, 15, 15],
            "scenario_id": [1, 1, 2, 1, 1, 2],
            "window_start_s": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "prob_0": [0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
            "prob_1": [0.1, 0.8, 0.7, 0.1, 0.2, 0.1],
            "prob_2": [0.1, 0.1, 0.2, 0.1, 0.7, 0.8],
        }
    )

    write_test_prediction_artifacts(
        output_dir=tmp_path,
        predictions=predictions,
        class_names=["still", "walk", "turn"],
        hop_size_s=1.0,
    )

    cms = tmp_path / "confusion_matrices"
    assert (cms / "test_all_subjects.html").exists()
    assert (cms / "test_subject_8.html").exists()
    assert (cms / "test_subject_15.html").exists()
    overview = cms / "test_subjects_overview.html"
    assert overview.exists()
    assert "macro-F1=" in overview.read_text(encoding="utf-8")

    assert (tmp_path / "plots" / "prediction_timeline_subject_8.html").exists()
    assert (tmp_path / "plots" / "prediction_timeline_subject_15.html").exists()

    subject_metrics = json.loads(
        (tmp_path / "reports" / "test_per_subject_metrics.json").read_text(encoding="utf-8")
    )
    assert [row["person_id"] for row in subject_metrics] == [8, 15]
    assert (tmp_path / "reports" / "test_per_subject_metrics.csv").exists()
