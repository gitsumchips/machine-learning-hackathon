"""
Evaluation script comparing Classifier-Only vs Classifier+HMM/Viterbi route reconstruction
across all 10 student-testing trips.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root is on path
sys.path.insert(0, os.path.abspath("."))

from src.hmm_decoder import decode_viterbi_route, compute_route_metrics, FEATURE_COLUMNS
from src.data_processing import load_trip

def main():
    print("=" * 65)
    print("PATHSNAP: HMM & VITERBI ROUTE RECONSTRUCTION EVALUATION")
    print("=" * 65)

    model_path = "models/pathsnap_hist_gradient_boosting_tuned.joblib"
    print(f"Loading tuned model from {model_path}...")
    model = joblib.load(model_path)

    test_parquet = "results/student_test_candidates_generated.parquet"
    print(f"Loading student test candidates from {test_parquet}...")
    df_test = pd.read_parquet(test_parquet)

    student_test_dir = Path("data/Student folder/student_test")
    test_trips = sorted([p.name for p in student_test_dir.iterdir() if p.is_dir()])
    print(f"Evaluating across {len(test_trips)} student-test trips...")

    results = []

    for trip_id in test_trips:
        trip_cands = df_test[df_test["trip_id"] == trip_id].copy().reset_index(drop=True)
        if len(trip_cands) == 0:
            continue

        raw_trip = load_trip(student_test_dir / trip_id, include_route=True)
        ref_route = raw_trip.route.tolist()
        nodes_df = raw_trip.nodes
        arcs_df = raw_trip.arcs

        # --- 1. Classifier-Only Prediction ---
        t0 = time.time()
        X = trip_cands[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        trip_cands["prob"] = probs

        # Pointwise top-1 arc
        top1_idx = trip_cands.groupby("observation_id")["prob"].idxmax()
        clf_pointwise_arcs = trip_cands.loc[top1_idx, "arc_id"].tolist()
        clf_pointwise_labels = trip_cands.loc[top1_idx, "label"].tolist()
        clf_top1_acc = float(np.mean(clf_pointwise_labels))

        # Collapse consecutive identical arcs for sequence comparison
        clf_collapsed = [clf_pointwise_arcs[0]]
        for a in clf_pointwise_arcs[1:]:
            if a != clf_collapsed[-1]:
                clf_collapsed.append(a)
        t_clf = time.time() - t0

        clf_metrics = compute_route_metrics(clf_collapsed, ref_route, arcs_df)

        # --- 2. Classifier + HMM / Viterbi Decoding ---
        t1 = time.time()
        viterbi_pointwise, viterbi_connected = decode_viterbi_route(
            candidate_df=trip_cands,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            model=model,
            switch_penalty=40.0,
            gap_penalty=80.0,
            gap_scale_m=50.0
        )
        t_hmm = time.time() - t1

        # Pointwise accuracy of Viterbi states against ground truth
        # Match back each observation's selected arc with target label
        obs_targets = trip_cands[trip_cands["label"] == 1].set_index("observation_id")["arc_id"].to_dict()
        hmm_correct = sum(
            1 for obs_idx, chosen_arc in enumerate(viterbi_pointwise)
            if obs_targets.get(obs_idx) == chosen_arc
        )
        hmm_top1_acc = float(hmm_correct / max(len(viterbi_pointwise), 1))

        hmm_metrics = compute_route_metrics(viterbi_connected, ref_route, arcs_df)

        print(f"\nTrip {trip_id} (Obs: {len(viterbi_pointwise)}, Ref Route: {len(ref_route)} arcs):")
        print(f"  Classifier-Only: Top-1 Acc: {clf_top1_acc:.4f} | Route Continuity: {clf_metrics['route_continuity']:.4f} | Route F1: {clf_metrics['arc_f1']:.4f} | Edit Sim: {clf_metrics['normalized_edit_similarity']:.4f}")
        print(f"  Classifier+HMM:  Top-1 Acc: {hmm_top1_acc:.4f} | Route Continuity: {hmm_metrics['route_continuity']:.4f} | Route F1: {hmm_metrics['arc_f1']:.4f} | Edit Sim: {hmm_metrics['normalized_edit_similarity']:.4f}")

        results.append({
            "trip_id": trip_id,
            "observations": len(viterbi_pointwise),
            "ref_route_length": len(ref_route),
            "clf_top1_accuracy": clf_top1_acc,
            "clf_route_continuity": clf_metrics["route_continuity"],
            "clf_route_arc_f1": clf_metrics["arc_f1"],
            "clf_edit_similarity": clf_metrics["normalized_edit_similarity"],
            "clf_time_seconds": t_clf,
            "hmm_top1_accuracy": hmm_top1_acc,
            "hmm_route_continuity": hmm_metrics["route_continuity"],
            "hmm_route_arc_f1": hmm_metrics["arc_f1"],
            "hmm_edit_similarity": hmm_metrics["normalized_edit_similarity"],
            "hmm_time_seconds": t_hmm
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("results/hmm_vs_classifier_comparison.csv", index=False)
    print("\nSaved detailed comparison to results/hmm_vs_classifier_comparison.csv")

    summary = {
        "evaluation_partition": "Student Testing (10 trips)",
        "total_trips": len(res_df),
        "mean_metrics": {
            "classifier_only": {
                "top1_accuracy": float(res_df["clf_top1_accuracy"].mean()),
                "route_continuity": float(res_df["clf_route_continuity"].mean()),
                "route_arc_f1": float(res_df["clf_route_arc_f1"].mean()),
                "route_edit_similarity": float(res_df["clf_edit_similarity"].mean()),
                "mean_time_seconds": float(res_df["clf_time_seconds"].mean())
            },
            "classifier_plus_hmm": {
                "top1_accuracy": float(res_df["hmm_top1_accuracy"].mean()),
                "route_continuity": float(res_df["hmm_route_continuity"].mean()),
                "route_arc_f1": float(res_df["hmm_route_arc_f1"].mean()),
                "route_edit_similarity": float(res_df["hmm_edit_similarity"].mean()),
                "mean_time_seconds": float(res_df["hmm_time_seconds"].mean())
            }
        },
        "improvements": {
            "route_continuity_gain": float(res_df["hmm_route_continuity"].mean() - res_df["clf_route_continuity"].mean()),
            "route_arc_f1_gain": float(res_df["hmm_route_arc_f1"].mean() - res_df["clf_route_arc_f1"].mean()),
            "edit_similarity_gain": float(res_df["hmm_edit_similarity"].mean() - res_df["clf_edit_similarity"].mean())
        }
    }

    with open("results/hmm_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("HMM VS CLASSIFIER-ONLY SUMMARY METRICS")
    print("=" * 65)
    print(f"Classifier-Only   -> Top-1 Acc: {summary['mean_metrics']['classifier_only']['top1_accuracy']:.4f} | Route Continuity: {summary['mean_metrics']['classifier_only']['route_continuity']*100:.2f}% | Route F1: {summary['mean_metrics']['classifier_only']['route_arc_f1']:.4f} | Edit Sim: {summary['mean_metrics']['classifier_only']['route_edit_similarity']:.4f}")
    print(f"Classifier + HMM  -> Top-1 Acc: {summary['mean_metrics']['classifier_plus_hmm']['top1_accuracy']:.4f} | Route Continuity: {summary['mean_metrics']['classifier_plus_hmm']['route_continuity']*100:.2f}% | Route F1: {summary['mean_metrics']['classifier_plus_hmm']['route_arc_f1']:.4f} | Edit Sim: {summary['mean_metrics']['classifier_plus_hmm']['route_edit_similarity']:.4f}")
    print(f"Continuity Improvement: +{summary['improvements']['route_continuity_gain']*100:.2f}%")
    print(f"Route Edit Similarity Improvement: +{summary['improvements']['edit_similarity_gain']:.4f}")

if __name__ == "__main__":
    main()
