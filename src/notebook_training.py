"""Training-only grouped cross-validation workflow for the PathSnap notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_training import CANDIDATE_FEATURES, generate_trip_candidates
from src.data_processing import iter_trip_directories


def _build_model_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_features = ["heading_valid"]
    numeric_features = [
        name for name in CANDIDATE_FEATURES if name not in categorical_features
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    return model, categorical_features


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    predictions = (probabilities >= 0.5).astype(np.int8)
    precision = float(
        precision_score(labels, predictions, average="binary", zero_division=0)
    )
    recall = float(
        recall_score(labels, predictions, average="binary", zero_division=0)
    )
    f1 = float(f1_score(labels, predictions, average="binary", zero_division=0))
    harmonic_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if not np.isclose(f1, harmonic_f1, rtol=0.0, atol=1e-12):
        raise ArithmeticError(
            f"Binary F1 {f1} does not match precision/recall harmonic mean {harmonic_f1}"
        )

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_harmonic_check": float(harmonic_f1),
        "f1_consistent": True,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "average": "binary",
        "threshold": 0.5,
        "candidate_rows": int(len(labels)),
    }


def run_notebook_training(
    train_dir: str | Path,
    *,
    model_path: str | Path,
    candidates_path: str | Path,
    cv_results_path: str | Path,
    oof_predictions_path: str | Path,
    summary_path: str | Path,
    top_k: int = 128,
    max_distance_m: float = 200.0,
    cell_size_m: float = 100.0,
    max_route_speed_mps: float = 70.0,
    start_tolerance_m: float = 200.0,
    end_tolerance_m: float = 600.0,
    advance_penalty_m: float = 5.0,
    n_splits: int = 5,
    random_seed: int = 42,
    expected_trip_count: int | None = 80,
) -> dict[str, Any]:
    """Generate train-only candidates, grouped OOF metrics, and the final model."""
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from sklearn.model_selection import GroupKFold

    training_dir = Path(train_dir).resolve()
    if training_dir.name.lower() != "train" or "student_test" in {
        part.lower() for part in training_dir.parts
    }:
        raise ValueError("Only results/features/train may be used for model fitting")
    if not training_dir.is_dir():
        raise FileNotFoundError(f"Training feature directory does not exist: {training_dir}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if top_k < 2:
        raise ValueError("top_k must be at least 2 to produce positive and negative rows")

    trip_dirs = list(iter_trip_directories(training_dir))
    if expected_trip_count is not None and len(trip_dirs) != expected_trip_count:
        raise ValueError(
            f"Expected {expected_trip_count} training trips, found {len(trip_dirs)}"
        )
    if len(trip_dirs) < n_splits:
        raise ValueError(
            f"GroupKFold({n_splits}) requires at least {n_splits} training trips"
        )

    candidate_file = Path(candidates_path)
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_candidate_file = candidate_file.with_name(candidate_file.name + ".tmp")
    if temporary_candidate_file.exists():
        temporary_candidate_file.unlink()

    generation_options = {
        "top_k": top_k,
        "max_distance_m": max_distance_m,
        "cell_size_m": cell_size_m,
        "max_route_speed_mps": max_route_speed_mps,
        "route_start_tolerance_m": start_tolerance_m,
        "route_end_tolerance_m": end_tolerance_m,
        "route_alignment_advance_penalty_m": advance_penalty_m,
    }
    trip_summaries: list[dict[str, Any]] = []
    writer = None
    candidate_row_count = 0
    try:
        for trip_dir in trip_dirs:
            candidates, trip_summary = generate_trip_candidates(
                trip_dir,
                top_k=top_k,
                max_distance_m=max_distance_m,
                cell_size_m=cell_size_m,
                max_route_speed_mps=max_route_speed_mps,
                start_tolerance_m=start_tolerance_m,
                end_tolerance_m=end_tolerance_m,
                advance_penalty_m=advance_penalty_m,
            )
            if candidates.empty:
                raise ValueError(
                    f"Trip {trip_dir.name} produced no labeled candidate rows"
                )
            if set(candidates["trip_id"]) != {trip_dir.name}:
                raise ValueError(f"Trip ID mismatch while processing {trip_dir.name}")
            if candidates["label"].isna().any() or not set(
                candidates["label"].unique()
            ).issubset({0, 1}):
                raise ValueError(f"Trip {trip_dir.name} has invalid binary labels")
            positive_per_observation = candidates.groupby(
                ["trip_id", "observation_id"], sort=False
            )["label"].sum()
            if not positive_per_observation.eq(1).all():
                raise ValueError(
                    f"Trip {trip_dir.name} has candidate observations without exactly "
                    "one positive label"
                )
            if candidates["label"].nunique() != 2:
                raise ValueError(
                    f"Trip {trip_dir.name} must include both positive and negative candidates"
                )

            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = parquet.ParquetWriter(temporary_candidate_file, table.schema)
            writer.write_table(table, row_group_size=64_000)
            candidate_row_count += len(candidates)
            trip_summaries.append(trip_summary)
    finally:
        if writer is not None:
            writer.close()

    if not trip_summaries:
        raise ValueError(f"No training trips were processed from {training_dir}")
    temporary_candidate_file.replace(candidate_file)
    parquet_file = parquet.ParquetFile(candidate_file)
    if parquet_file.metadata.num_rows != candidate_row_count:
        raise IOError("Candidate Parquet row count does not match generated rows")

    trip_ids = np.asarray([str(item["trip_id"]) for item in trip_summaries])
    if len(np.unique(trip_ids)) != len(trip_ids):
        raise ValueError("Training trip IDs must be unique")
    if n_splits > len(trip_ids):
        raise ValueError("n_splits cannot exceed the number of training trips")

    group_kfold = GroupKFold(n_splits=n_splits)
    fold_splits = list(
        group_kfold.split(np.zeros((len(trip_ids), 1)), groups=trip_ids)
    )
    output_cv_path = Path(cv_results_path)
    output_cv_path.parent.mkdir(parents=True, exist_ok=True)
    output_oof_path = Path(oof_predictions_path)
    output_oof_path.parent.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    oof_labels: list[np.ndarray] = []
    oof_probabilities: list[np.ndarray] = []
    oof_file_written = False
    metadata_columns = ["trip_id", "observation_id", "arc_id", "label"]

    for fold_number, (train_indices, validation_indices) in enumerate(
        fold_splits, start=1
    ):
        train_trip_ids = trip_ids[train_indices].tolist()
        validation_trip_ids = trip_ids[validation_indices].tolist()
        if set(train_trip_ids) & set(validation_trip_ids):
            raise RuntimeError(f"Trip leakage detected in fold {fold_number}")

        training_candidates = pd.read_parquet(
            candidate_file,
            engine="pyarrow",
            filters=[("trip_id", "in", train_trip_ids)],
        )
        validation_candidates = pd.read_parquet(
            candidate_file,
            engine="pyarrow",
            filters=[("trip_id", "in", validation_trip_ids)],
        )
        if training_candidates.empty or validation_candidates.empty:
            raise ValueError(f"Fold {fold_number} has an empty partition")

        features = [name for name in CANDIDATE_FEATURES if name in training_candidates]
        forbidden = {
            "label",
            "trip_id",
            "observation_id",
            "arc_id",
            "route_position",
            "route_arc_id",
        }
        if set(features) & forbidden or set(features) != set(CANDIDATE_FEATURES):
            raise ValueError("Model features include target/IDs or omit candidate features")

        model, categorical_features = _build_model_pipeline()
        training_labels = training_candidates["label"].to_numpy(dtype=np.int8)
        validation_labels = validation_candidates["label"].to_numpy(dtype=np.int8)
        if len(np.unique(training_labels)) != 2 or len(np.unique(validation_labels)) != 2:
            raise ValueError(f"Fold {fold_number} does not contain both binary labels")
        model.set_params(classifier__random_state=random_seed)
        model.fit(training_candidates[features], training_labels)
        probabilities = model.predict_proba(validation_candidates[features])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int8)

        fold_metrics = _binary_metrics(validation_labels, probabilities)
        fold_results.append(
            {
                "fold": fold_number,
                "training_trip_count": len(train_trip_ids),
                "validation_trip_count": len(validation_trip_ids),
                "training_candidate_rows": int(len(training_candidates)),
                "validation_candidate_rows": int(len(validation_candidates)),
                "validation_trip_ids": ";".join(validation_trip_ids),
                **fold_metrics,
            }
        )
        oof_labels.append(validation_labels)
        oof_probabilities.append(probabilities.astype(np.float32, copy=False))

        predictions_frame = validation_candidates[metadata_columns].copy()
        predictions_frame["fold"] = fold_number
        predictions_frame["probability_positive"] = probabilities
        predictions_frame["prediction"] = predictions
        predictions_frame.to_csv(
            output_oof_path,
            mode="a" if oof_file_written else "w",
            header=not oof_file_written,
            index=False,
        )
        oof_file_written = True

    all_oof_labels = np.concatenate(oof_labels)
    all_oof_probabilities = np.concatenate(oof_probabilities)
    overall_metrics = _binary_metrics(all_oof_labels, all_oof_probabilities)
    cv_rows = [
        {
            key: value
            for key, value in fold.items()
            if key != "confusion_matrix"
        }
        | {
            f"confusion_{key}": value
            for key, value in fold["confusion_matrix"].items()
        }
        for fold in fold_results
    ]
    cv_rows.append(
        {
            "fold": "OOF",
            "training_trip_count": len(trip_ids),
            "validation_trip_count": len(trip_ids),
            "training_candidate_rows": candidate_row_count,
            "validation_candidate_rows": len(all_oof_labels),
            "validation_trip_ids": ";".join(trip_ids.tolist()),
            **{
                key: value
                for key, value in overall_metrics.items()
                if key != "confusion_matrix"
            },
            **{
                f"confusion_{key}": value
                for key, value in overall_metrics["confusion_matrix"].items()
            },
        }
    )
    pd.DataFrame(cv_rows).to_csv(output_cv_path, index=False)

    all_candidates = pd.read_parquet(candidate_file, engine="pyarrow")
    final_model, categorical_features = _build_model_pipeline()
    final_model.set_params(classifier__random_state=random_seed)
    final_labels = all_candidates["label"].to_numpy(dtype=np.int8)
    final_features = [name for name in CANDIDATE_FEATURES if name in all_candidates]
    final_model.fit(all_candidates[final_features], final_labels)

    final_model_path = Path(model_path)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": final_features,
            "categorical_features": categorical_features,
            "candidate_generation": generation_options,
            "training_trip_ids": trip_ids.tolist(),
        },
        final_model_path,
    )

    summary = {
        "training_source": str(training_dir),
        "training_trip_count": len(trip_ids),
        "training_trip_ids": trip_ids.tolist(),
        "candidate_generation": generation_options,
        "candidate_rows": candidate_row_count,
        "positive_candidates": int(final_labels.sum()),
        "negative_candidates": int(len(final_labels) - final_labels.sum()),
        "candidate_recall_pct_all_training_trips": 100.0
        * sum(item["candidate_hit_observations"] for item in trip_summaries)
        / sum(item["observations"] for item in trip_summaries),
        "trip_summaries": trip_summaries,
        "validation": {
            "strategy": "5-fold GroupKFold by trip_id",
            "fold_count": n_splits,
            "metric_average": "binary",
            "metrics_on_pooled_out_of_fold_predictions": overall_metrics,
            "fold_metrics": fold_results,
            "folds_disjoint_by_trip": True,
        },
        "features": {
            "numeric_and_categorical": final_features,
            "categorical_one_hot": categorical_features,
            "excluded_columns": [
                "label",
                "trip_id",
                "observation_id",
                "arc_id",
                "reference_route",
                "route_position",
            ],
            "preprocessing_fit_within_each_fold": True,
        },
        "outputs": {
            "candidate_parquet": str(candidate_file),
            "cv_results_csv": str(output_cv_path),
            "oof_predictions_csv": str(output_oof_path),
            "model": str(final_model_path),
            "summary_json": str(Path(summary_path)),
        },
    }
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
