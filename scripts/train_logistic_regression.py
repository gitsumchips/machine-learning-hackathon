"""Generate route-labeled road candidates and train Logistic Regression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.notebook_training import run_notebook_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run training-only candidate generation and 5-fold trip-grouped Logistic Regression."
    )
    parser.add_argument("--train-dir", default="results/features/train")
    parser.add_argument(
        "--model-path", default="models/logistic_regression_candidate.joblib"
    )
    parser.add_argument(
        "--candidates-path", default="results/train_candidates_generated.parquet"
    )
    parser.add_argument(
        "--cv-results-path", default="results/logistic_regression_cv_results.csv"
    )
    parser.add_argument(
        "--oof-predictions-path",
        default="results/logistic_regression_oof_predictions.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="results/logistic_regression_training_summary_notebook.json",
    )
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--max-distance-m", type=float, default=200.0)
    parser.add_argument("--cell-size-m", type=float, default=100.0)
    parser.add_argument("--max-route-speed-mps", type=float, default=70.0)
    parser.add_argument("--start-tolerance-m", type=float, default=200.0)
    parser.add_argument("--end-tolerance-m", type=float, default=600.0)
    parser.add_argument("--advance-penalty-m", type=float, default=5.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_notebook_training(
        args.train_dir,
        model_path=args.model_path,
        candidates_path=args.candidates_path,
        cv_results_path=args.cv_results_path,
        oof_predictions_path=args.oof_predictions_path,
        summary_path=args.summary_path,
        top_k=args.top_k,
        max_distance_m=args.max_distance_m,
        cell_size_m=args.cell_size_m,
        max_route_speed_mps=args.max_route_speed_mps,
        start_tolerance_m=args.start_tolerance_m,
        end_tolerance_m=args.end_tolerance_m,
        advance_penalty_m=args.advance_penalty_m,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
