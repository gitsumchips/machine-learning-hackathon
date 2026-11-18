import numpy as np
import pandas as pd
import pytest

from src.candidate_training import (
    CANDIDATE_FEATURES,
    SegmentGridIndex,
    _read_trip_feature_tables,
    align_route_to_track,
    generate_trip_candidates,
    train_logistic_regression,
)
from src.notebook_training import run_notebook_training


def test_segment_grid_finds_candidates_in_long_segment_interiors():
    index = SegmentGridIndex(
        np.array([3]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1000.0]),
        np.array([0.0]),
        cell_size_m=10.0,
    )

    arc_ids, row_ids, distances, _ = index.query(
        250.0, 0.0, top_k=1, max_distance_m=20.0
    )

    np.testing.assert_array_equal(arc_ids, [3])
    np.testing.assert_array_equal(row_ids, [0])
    np.testing.assert_allclose(distances, [0.0])


def test_route_alignment_is_chronological_for_a_simple_path():
    aligned = align_route_to_track(
        track_x=np.array([1.0, 15.0, 29.0]),
        track_y=np.zeros(3),
        route_arc_ids=np.array([10, 11, 12]),
        arc_ids=np.array([10, 11, 12]),
        arc_start_x=np.array([0.0, 10.0, 20.0]),
        arc_start_y=np.zeros(3),
        arc_end_x=np.array([10.0, 20.0, 30.0]),
        arc_end_y=np.zeros(3),
    )

    np.testing.assert_array_equal(aligned, [10, 11, 12])


def test_route_alignment_rejects_impossible_route_jump_for_elapsed_time():
    aligned = align_route_to_track(
        track_x=np.array([1.0, 1001.0]),
        track_y=np.zeros(2),
        route_arc_ids=np.array([10, 11]),
        arc_ids=np.array([10, 11]),
        arc_start_x=np.array([0.0, 1000.0]),
        arc_start_y=np.zeros(2),
        arc_end_x=np.array([10.0, 1010.0]),
        arc_end_y=np.zeros(2),
        track_elapsed_time_s=np.array([0.0, 1.0]),
        max_route_speed_mps=70.0,
        end_tolerance_m=1000.0,
    )

    np.testing.assert_array_equal(aligned, [10, 10])


def _write_synthetic_trip(trip_dir):
    trip_dir.mkdir()
    pd.DataFrame(
        {
            "observation_id": [0, 1],
            "longitude": [0.0002, 0.0008],
            "latitude": [0.00001, 0.00001],
            "elapsed_time_s": [0.0, 1.0],
            "delta_time_s": [0.0, 1.0],
            "step_distance_m": [0.0, 66.8],
            "speed_mps": [0.0, 66.8],
            "acceleration_mps2": [0.0, 66.8],
            "heading_sin": [0.0, 1.0],
            "heading_cos": [0.0, 0.0],
            "heading_valid": [0, 1],
        }
    ).to_csv(trip_dir / "trajectory_features.csv", index=False)
    pd.DataFrame(
        {
            "arc_id": [0, 1],
            "from_node_id": [0, 2],
            "to_node_id": [1, 3],
            "from_longitude": [0.0, 0.0],
            "from_latitude": [0.0, 0.0001],
            "to_longitude": [0.001, 0.001],
            "to_latitude": [0.0, 0.0001],
            "midpoint_longitude": [0.0005, 0.0005],
            "midpoint_latitude": [0.0, 0.0001],
            "length_m": [111.2, 111.2],
            "bearing_sin": [1.0, 1.0],
            "bearing_cos": [0.0, 0.0],
            "from_out_degree": [1, 1],
            "to_in_degree": [1, 1],
        }
    ).to_csv(trip_dir / "road_segment_features.csv", index=False)
    pd.DataFrame({"route_position": [0], "arc_id": [0]}).to_csv(
        trip_dir / "route_labels.csv", index=False
    )


def test_candidate_generation_builds_binary_labeled_observation_arc_rows(tmp_path):
    trip_dir = tmp_path / "synthetic_trip"
    _write_synthetic_trip(trip_dir)

    candidates, summary = generate_trip_candidates(
        trip_dir,
        top_k=2,
        max_distance_m=30.0,
        cell_size_m=50.0,
    )

    assert set(candidates["trip_id"]) == {"synthetic_trip"}
    assert len(candidates) == 4
    assert set(candidates["label"].unique()) == {0, 1}
    assert candidates.groupby("observation_id")["label"].sum().tolist() == [1, 1]
    assert set(CANDIDATE_FEATURES).issubset(candidates.columns)
    assert summary["candidate_recall_pct"] == 100.0


def test_candidate_generation_excludes_observations_without_a_positive(tmp_path):
    trip_dir = tmp_path / "synthetic_trip"
    _write_synthetic_trip(trip_dir)
    trajectory_path = trip_dir / "trajectory_features.csv"
    trajectory = pd.read_csv(trajectory_path)
    trajectory.loc[1, "longitude"] = 0.02
    trajectory.to_csv(trajectory_path, index=False)

    candidates, summary = generate_trip_candidates(
        trip_dir,
        top_k=2,
        max_distance_m=30.0,
        cell_size_m=50.0,
    )

    assert candidates["observation_id"].unique().tolist() == [0]
    assert candidates.groupby("observation_id")["label"].sum().tolist() == [1]
    assert summary["candidate_recall_pct"] == 50.0
    assert summary["excluded_unmatched_observations"] == 1


def test_feature_tables_sort_routes_and_validate_timestamp_order(tmp_path):
    trip_dir = tmp_path / "synthetic_trip"
    _write_synthetic_trip(trip_dir)

    route_path = trip_dir / "route_labels.csv"
    pd.DataFrame(
        {"route_position": [1, 0], "arc_id": [1, 0]}
    ).to_csv(route_path, index=False)
    _, roads, route_ids = _read_trip_feature_tables(trip_dir)
    np.testing.assert_array_equal(roads["arc_id"], [0, 1])
    np.testing.assert_array_equal(route_ids, [0, 1])

    trajectory_path = trip_dir / "trajectory_features.csv"
    trajectory = pd.read_csv(trajectory_path)
    trajectory.loc[1, "delta_time_s"] = 0.0
    trajectory.to_csv(trajectory_path, index=False)
    with pytest.raises(ValueError, match="timestamps are not chronological"):
        _read_trip_feature_tables(trip_dir)


def test_logistic_training_splits_by_trip_and_saves_pipeline(tmp_path):
    rows = []
    for trip_idx in range(4):
        for obs_idx in range(3):
            for label in (0, 1):
                row = {name: float(label + obs_idx * 0.01) for name in CANDIDATE_FEATURES}
                row.update(
                    trip_id=f"trip_{trip_idx}",
                    observation_id=obs_idx,
                    arc_id=label,
                    label=label,
                )
                rows.append(row)
    candidates = pd.DataFrame(rows)
    model_path = tmp_path / "model.joblib"

    summary = train_logistic_regression(
        candidates, model_path=model_path, validation_size=0.5
    )

    assert model_path.is_file()
    assert len(summary["validation_trip_ids"]) == 2
    assert summary["training_trip_count"] == 4
    assert 0.0 <= summary["validation_metrics"]["candidate_f1"] <= 1.0
    metrics = summary["validation_metrics"]
    expected_f1 = (
        2 * metrics["candidate_precision"] * metrics["candidate_recall"]
        / (metrics["candidate_precision"] + metrics["candidate_recall"])
        if metrics["candidate_precision"] + metrics["candidate_recall"]
        else 0.0
    )
    assert metrics["candidate_metric_average"] == "binary"
    assert metrics["candidate_f1"] == pytest.approx(expected_f1)


def test_notebook_training_runs_grouped_cv_and_writes_expected_outputs(tmp_path):
    train_dir = tmp_path / "results" / "features" / "train"
    train_dir.mkdir(parents=True)
    for trip_idx in range(5):
        _write_synthetic_trip(train_dir / f"trip_{trip_idx}")

    outputs = {
        "model_path": tmp_path / "models" / "model.joblib",
        "candidates_path": tmp_path / "results" / "train_candidates_generated.parquet",
        "cv_results_path": tmp_path / "results" / "cv_results.csv",
        "oof_predictions_path": tmp_path / "results" / "oof_predictions.csv",
        "summary_path": tmp_path / "results" / "training_summary.json",
    }
    summary = run_notebook_training(
        train_dir,
        **outputs,
        top_k=2,
        max_distance_m=30.0,
        cell_size_m=50.0,
        n_splits=5,
        expected_trip_count=5,
    )

    assert all(path.is_file() for path in outputs.values())
    assert summary["training_trip_count"] == 5
    assert summary["positive_candidates"] == summary["negative_candidates"]
    assert len(summary["validation"]["fold_metrics"]) == 5
    assert summary["validation"]["folds_disjoint_by_trip"]
    assert summary["validation"]["metrics_on_pooled_out_of_fold_predictions"][
        "f1_consistent"
    ]
