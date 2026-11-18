"""Random Forest candidate ranker for GPS-to-road map matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import iter_trip_directories, load_trip
from src.knn_model import (
    FEATURE_COLUMNS,
    evaluate_route_sequences,
    make_dataset,
    rank_candidates,
    tune_route_decoder,
)


def build_random_forest(
    *,
    n_estimators: int = 150,
    max_features: str | float = "sqrt",
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> object:
    """Create a reproducible, class-balanced Random Forest pipeline."""
    if n_estimators < 1 or min_samples_leaf < 1:
        raise ValueError("n_estimators and min_samples_leaf must be positive")
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    # Keep a pipeline interface compatible with the shared route decoder.
    return make_pipeline(classifier)


def tune_random_forest(
    training_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    *,
    configurations: Iterable[dict[str, object]],
    random_state: int = 42,
) -> tuple[object, pd.DataFrame]:
    """Compare forest configurations by top-ranked proxy-arc accuracy."""
    configurations = list(configurations)
    if training_examples.empty or validation_examples.empty:
        raise ValueError("Training and validation candidate tables must be non-empty")
    if not configurations:
        raise ValueError("At least one Random Forest configuration is required")

    x_train = training_examples[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = training_examples["label"].to_numpy(dtype=np.int8)
    targets = validation_examples[
        ["trip_id", "observation_id", "target_arc_id"]
    ].drop_duplicates()
    results = []
    models = []
    for config in configurations:
        model = build_random_forest(random_state=random_state, **config)
        model.fit(x_train, y_train)
        ranked = rank_candidates(model, validation_examples)
        checked = ranked.merge(
            targets,
            on=["trip_id", "observation_id"],
            validate="one_to_one",
        )
        score = float(np.mean(checked["arc_id"].to_numpy() == checked["target_arc_id"].to_numpy()))
        result = dict(config)
        result["proxy_top1_accuracy"] = score
        results.append(result)
        models.append(model)

    result_frame = pd.DataFrame(results).sort_values(
        ["proxy_top1_accuracy", "n_estimators"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    best_position = int(result_frame.index[0])
    # Keep returned best model in lockstep with the ranked result rows.
    best_config = result_frame.iloc[0]
    best_model = models[
        next(
            index
            for index, config in enumerate(configurations)
            if all(best_config[key] == value for key, value in config.items())
        )
    ]
    return best_model, result_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/Student folder")
    parser.add_argument("--output", default="results/random_forest/pathsnap_random_forest.joblib")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-observations", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--max-eval-observations", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller forest and a single configuration for a quick smoke test.",
    )
    args = parser.parse_args()

    if not 0 < args.validation_size < 1 or not 0 < args.test_size < 1:
        raise ValueError("--validation-size and --test-size must be between 0 and 1")
    if args.validation_size + args.test_size >= 1:
        raise ValueError("--validation-size plus --test-size must be less than 1")

    train_dirs = list(iter_trip_directories(Path(args.data_root) / "train"))
    train_ids = [directory.name for directory in train_dirs]
    if len(train_ids) < 3:
        raise ValueError("At least three training trips are required for fit/validation/test splits")
    fit_validation_ids, test_ids = train_test_split(
        train_ids, test_size=args.test_size, random_state=args.random_state
    )
    fit_ids, validation_ids = train_test_split(
        fit_validation_ids,
        test_size=args.validation_size,
        random_state=args.random_state + 1,
    )
    paths_by_id = {directory.name: directory for directory in train_dirs}
    fit_trips = [load_trip(paths_by_id[trip_id], include_route=True) for trip_id in fit_ids]
    validation_trips = [
        load_trip(paths_by_id[trip_id], include_route=True) for trip_id in validation_ids
    ]
    test_trips = [load_trip(paths_by_id[trip_id], include_route=True) for trip_id in test_ids]
    data_options = {
        "max_observations": args.max_observations,
        "max_candidates": args.max_candidates,
        "max_negatives": args.max_negatives,
    }
    fit_examples = make_dataset(fit_trips, training=True, **data_options)
    validation_examples = make_dataset(validation_trips, training=True, **data_options)

    tree_count = min(args.n_estimators, 40) if args.quick else args.n_estimators
    if args.quick:
        configurations = [
            {"n_estimators": tree_count, "max_features": "sqrt", "min_samples_leaf": 1}
        ]
    else:
        configurations = [
            {"n_estimators": tree_count, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": tree_count, "max_features": "sqrt", "min_samples_leaf": 3},
            {"n_estimators": tree_count, "max_features": 0.7, "min_samples_leaf": 1},
        ]

    tuned_model, tuning_results = tune_random_forest(
        fit_examples,
        validation_examples,
        configurations=configurations,
        random_state=args.random_state,
    )
    selected = tuning_results.iloc[0]
    selected_config = {
        key: selected[key]
        for key in ("n_estimators", "max_features", "min_samples_leaf")
    }
    # DataFrame serialization may return NumPy scalar values; normalize them.
    selected_config["n_estimators"] = int(selected_config["n_estimators"])
    selected_config["min_samples_leaf"] = int(selected_config["min_samples_leaf"])
    if isinstance(selected_config["max_features"], np.generic):
        selected_config["max_features"] = selected_config["max_features"].item()

    decoder_results = tune_route_decoder(
        tuned_model,
        validation_trips,
        max_candidates=args.max_candidates,
        max_observations=args.max_eval_observations,
    )
    switch_penalty = float(decoder_results.iloc[0]["switch_penalty"])
    heldout_metrics, per_trip_metrics = evaluate_route_sequences(
        tuned_model,
        test_trips,
        max_candidates=args.max_candidates,
        switch_penalty=switch_penalty,
        max_observations=args.max_eval_observations,
    )

    # Final deployment model is refit on every labeled training trip only after
    # configuration selection and held-out evaluation are finished.
    all_train_examples = make_dataset(fit_trips + validation_trips + test_trips, training=True, **data_options)
    final_model = build_random_forest(
        **selected_config,
        random_state=args.random_state,
    )
    final_model.fit(
        all_train_examples[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        all_train_examples["label"].to_numpy(dtype=np.int8),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "model_type": "RandomForestClassifier",
            "hyperparameters": selected_config,
            "max_candidates": args.max_candidates,
            "switch_penalty": switch_penalty,
            "training_label_definition": "nearest reference-route arc by point-to-segment distance",
            "heldout_test_route_metrics": heldout_metrics,
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    report = {
        "selected_hyperparameters": selected_config,
        "validation_candidate_tuning": tuning_results.to_dict(orient="records"),
        "validation_route_decoder_tuning": decoder_results.to_dict(orient="records"),
        "heldout_test_route_metrics": heldout_metrics,
        "heldout_test_per_trip": per_trip_metrics.to_dict(orient="records"),
        "training_label_note": "Training uses per-observation proxy labels derived by nearest point-to-segment distance among reference-route arcs.",
        "evaluation_note": "Held-out routes are decoded without route labels in candidate generation, then compared with each trip's supplied .route sequence.",
        "trip_counts": {"fit": len(fit_trips), "validation": len(validation_trips), "heldout_test": len(test_trips)},
    }
    metrics_path.write_text(json.dumps(report, indent=2))

    print(f"Training candidate rows: {len(fit_examples):,}")
    print(f"Validation candidate rows: {len(validation_examples):,}")
    print("Random Forest tuning (validation proxy top-1):")
    print(tuning_results.to_string(index=False))
    print("Route decoder tuning:")
    print(decoder_results.to_string(index=False))
    print("Held-out route-sequence metrics against .route:")
    for metric, value in heldout_metrics.items():
        print(f"  {metric}: {value}")
    print(f"Saved model: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
