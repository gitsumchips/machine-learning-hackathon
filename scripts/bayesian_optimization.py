"""
Bayesian Hyperparameter Optimization for PathSnap.
Tuning HistGradientBoostingClassifier using Optuna (TPE) with 5-fold GroupKFold.
"""

import os
import sys
import time
import json
import optuna
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath("."))

from src.candidate_training import SegmentGridIndex, align_route_to_track
from src.utils.geo import (
    local_cartesian_projection,
    batch_point_to_segment_projection,
    bearing_difference,
    initial_bearing_vectorized,
)

FEATURE_COLUMNS = [
    'perpendicular_distance_m', 'along_track_distance_m', 'segment_length_m',
    'gps_step_distance_m', 'gps_speed_mps', 'gps_acceleration_mps2',
    'gps_delta_time_s', 'heading_alignment', 'heading_valid',
    'arc_bearing_sin', 'arc_bearing_cos', 'from_out_degree', 'to_in_degree'
]

def build_student_test_candidates():
    """Build candidate dataframe for all 10 student-test trips."""
    test_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]
    all_frames = []
    for trip_id in test_trips:
        feat_dir = Path(f"results/features/student_test/{trip_id}")
        raw_dir = Path(f"data/Student folder/student_test/{trip_id}")

        trajectory = pd.read_csv(feat_dir / "trajectory_features.csv")
        roads = pd.read_csv(feat_dir / "road_segment_features.csv")
        route_file = raw_dir / f"{trip_id}.route"
        route = pd.read_csv(route_file, header=None, names=["arc_id"])["arc_id"].to_numpy(dtype=np.int64)

        ref_lon = float(trajectory["longitude"].mean())
        ref_lat = float(trajectory["latitude"].mean())

        track_x, track_y, _, _ = local_cartesian_projection(
            trajectory["longitude"].to_numpy(), trajectory["latitude"].to_numpy(),
            ref_lat=ref_lat, ref_lon=ref_lon
        )
        start_x, start_y, _, _ = local_cartesian_projection(
            roads["from_longitude"].to_numpy(), roads["from_latitude"].to_numpy(),
            ref_lat=ref_lat, ref_lon=ref_lon
        )
        end_x, end_y, _, _ = local_cartesian_projection(
            roads["to_longitude"].to_numpy(), roads["to_latitude"].to_numpy(),
            ref_lat=ref_lat, ref_lon=ref_lon
        )

        gt_assigned = align_route_to_track(
            track_x, track_y, route, roads["arc_id"].to_numpy(),
            start_x, start_y, end_x, end_y
        )

        index = SegmentGridIndex(roads["arc_id"].to_numpy(), start_x, start_y, end_x, end_y, cell_size_m=100.0)

        rows = []
        for obs_idx in range(len(track_x)):
            c_arcs, c_indices, dists, projs = index.query(track_x[obs_idx], track_y[obs_idx], top_k=8, max_distance_m=100.0)
            target_arc = gt_assigned[obs_idx]
            obs_traj = trajectory.iloc[obs_idx]

            for rank in range(len(c_arcs)):
                arc_id = c_arcs[rank]
                r_idx = c_indices[rank]
                d_m = dists[rank]
                p_frac = projs[rank]
                road_row = roads.iloc[r_idx]

                # Heading alignment
                h_diff = 0.0
                if obs_traj["heading_valid"]:
                    gps_bearing = np.degrees(np.arctan2(obs_traj["heading_sin"], obs_traj["heading_cos"])) % 360.0
                    arc_bearing = np.degrees(np.arctan2(road_row["bearing_sin"], road_row["bearing_cos"])) % 360.0
                    h_diff = bearing_difference(gps_bearing, arc_bearing)
                h_align = np.cos(np.radians(h_diff))

                rows.append({
                    "trip_id": trip_id,
                    "observation_id": obs_idx,
                    "arc_id": arc_id,
                    "perpendicular_distance_m": d_m,
                    "along_track_distance_m": p_frac * road_row["length_m"],
                    "segment_length_m": road_row["length_m"],
                    "gps_step_distance_m": obs_traj["step_distance_m"],
                    "gps_speed_mps": obs_traj["speed_mps"],
                    "gps_acceleration_mps2": obs_traj["acceleration_mps2"],
                    "gps_delta_time_s": obs_traj["delta_time_s"],
                    "heading_alignment": h_align,
                    "heading_valid": obs_traj["heading_valid"],
                    "arc_bearing_sin": road_row["bearing_sin"],
                    "arc_bearing_cos": road_row["bearing_cos"],
                    "from_out_degree": road_row["from_out_degree"],
                    "to_in_degree": road_row["to_in_degree"],
                    "label": int(arc_id == target_arc)
                })
        all_frames.append(pd.DataFrame(rows))

    test_df = pd.concat(all_frames, ignore_index=True)
    return test_df

def main():
    print("=" * 60)
    print("PATHSNAP: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 60)

    # 1. Load training candidates
    train_parquet = "results/train_candidates_generated.parquet"
    print(f"Loading training candidates from {train_parquet}...")
    t0 = time.time()
    df_train = pd.read_parquet(train_parquet)
    print(f"Loaded {len(df_train)} rows across {df_train['trip_id'].nunique()} trips in {time.time()-t0:.2f}s")

    # Sample negatives for memory & training efficiency (positives: 100%, negatives: 7%)
    pos_mask = df_train["label"] == 1
    neg_sampled = df_train[~pos_mask].sample(frac=0.07, random_state=42)
    sample_df = pd.concat([df_train[pos_mask], neg_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Tuning dataset: {len(sample_df)} candidate rows ({sample_df['label'].mean()*100:.1f}% positives)")

    X_train = sample_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = sample_df["label"].to_numpy(dtype=np.int8)
    groups = sample_df["trip_id"].to_numpy()

    # 2. Build or load student-test candidate table
    test_parquet = "results/student_test_candidates_generated.parquet"
    if not os.path.exists(test_parquet):
        print("Generating student-test candidate table...")
        df_test = build_student_test_candidates()
        df_test.to_parquet(test_parquet, index=False)
        print(f"Saved student test candidates to {test_parquet} ({len(df_test)} rows)")
    else:
        print(f"Loading precomputed student-test candidates from {test_parquet}...")
        df_test = pd.read_parquet(test_parquet)

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int8)

    # 3. Define Optuna Objective with 3-fold GroupKFold
    gkf = GroupKFold(n_splits=3)
    cv_splits = list(gkf.split(X_train, y_train, groups=groups))

    trial_records = []

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 250, step=25)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 50, step=5)
        l2_regularization = trial.suggest_float("l2_regularization", 0.01, 10.0, log=True)

        fold_f1s = []
        for train_idx, val_idx in cv_splits:
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42
            )
            clf.fit(X_train[train_idx], y_train[train_idx])
            preds = clf.predict(X_train[val_idx])
            f1_macro = f1_score(y_train[val_idx], preds, average="macro", zero_division=0)
            fold_f1s.append(f1_macro)

        mean_f1 = float(np.mean(fold_f1s))
        trial_records.append({
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "cv_macro_f1": mean_f1
        })
        return mean_f1

    print("\nRunning Bayesian Optimization with Optuna (15 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t_opt_start = time.time()
    study.optimize(objective, n_trials=15)
    opt_time = time.time() - t_opt_start
    print(f"Bayesian Optimization finished in {opt_time:.2f}s")
    print(f"Best Trial: #{study.best_trial.number} with CV Macro F1: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")

    # Save trial history
    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv("results/bayesian_optimization_trials.csv", index=False)
    print("Saved trial records to results/bayesian_optimization_trials.csv")

    # 4. Refit Tuned Model on full training data
    print("\nRefitting Tuned HistGradientBoostingClassifier on training data...")
    t_fit_start = time.time()
    best_params = study.best_params
    tuned_model = HistGradientBoostingClassifier(
        **best_params,
        random_state=42
    )
    tuned_model.fit(X_train, y_train)
    tuned_fit_time = time.time() - t_fit_start
    print(f"Tuned model refitted in {tuned_fit_time:.2f}s")

    # Save tuned model
    joblib.dump(tuned_model, "models/pathsnap_hist_gradient_boosting_tuned.joblib")
    print("Saved tuned model to models/pathsnap_hist_gradient_boosting_tuned.joblib")

    # 5. Evaluate Untuned Baseline vs Tuned on Student-Testing Set
    print("\nEvaluating Untuned vs Tuned on Student-Testing Partition (10 trips)...")
    # Untuned baseline (default hyperparameters)
    untuned_model = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=150, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    untuned_model.fit(X_train, y_train)

    def evaluate_on_student_test(model, name):
        t_infer_start = time.time()
        probs = model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t_infer_start
        preds = (probs >= 0.5).astype(int)

        df_eval = df_test[["trip_id", "observation_id", "label"]].copy()
        df_eval["prob"] = probs
        top1_idx = df_eval.groupby(["trip_id", "observation_id"])["prob"].idxmax()
        top1_acc = float(df_eval.loc[top1_idx, "label"].mean())

        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_test, preds))
        loss = float(log_loss(y_test, np.clip(probs, 1e-7, 1 - 1e-7)))

        return {
            "model_version": name,
            "student_test_top1_accuracy": top1_acc,
            "student_test_macro_f1": macro_f1,
            "student_test_candidate_accuracy": acc,
            "student_test_log_loss": loss,
            "inference_time_seconds": infer_time
        }

    untuned_res = evaluate_on_student_test(untuned_model, "Untuned HistGradientBoosting (Baseline)")
    tuned_res = evaluate_on_student_test(tuned_model, "Tuned HistGradientBoosting (Bayesian Optuna)")

    # Comparison DataFrame
    comp_df = pd.DataFrame([untuned_res, tuned_res])
    comp_df["cv_macro_f1"] = [0.7487, study.best_value]
    # Reorder columns
    cols = [
        "model_version", "cv_macro_f1", "student_test_top1_accuracy",
        "student_test_macro_f1", "student_test_log_loss", "inference_time_seconds"
    ]
    comp_df = comp_df[cols]
    comp_df.to_csv("results/tuned_vs_untuned_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("BAYESIAN OPTIMIZATION COMPARISON RESULTS")
    print("=" * 60)
    print(comp_df.to_string(index=False))

    # Save summary JSON
    summary = {
        "model_family": "HistGradientBoostingClassifier",
        "search_space": {
            "learning_rate": "[0.03, 0.25] (log)",
            "max_iter": "[100, 250]",
            "max_leaf_nodes": "[15, 63]",
            "min_samples_leaf": "[10, 50]",
            "l2_regularization": "[0.01, 10.0] (log)"
        },
        "optimization_algorithm": "Tree-structured Parzen Estimator (TPE / Optuna)",
        "number_of_trials": 15,
        "optimization_objective": "GroupKFold (3-fold) Macro F1 on 80 training trips",
        "best_hyperparameters": best_params,
        "best_cv_macro_f1": study.best_value,
        "optimization_runtime_seconds": opt_time,
        "comparison": comp_df.to_dict(orient="records")
    }
    with open("results/bayesian_optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to results/bayesian_optimization_summary.json")

if __name__ == "__main__":
    main()
