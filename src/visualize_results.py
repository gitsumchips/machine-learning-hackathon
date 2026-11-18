"""Generate EDA, saved-experiment comparisons, and maps without retraining.

This script only loads the saved KNN/Random Forest estimators for inference.
Cross-validation scores are not fabricated: the current saved reports contain
one fixed trip-level holdout, so CV macro-F1 would require new model fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import audit_split, iter_trip_directories, load_trip
from src.knn_model import (
    FEATURE_COLUMNS,
    decode_connected_route,
    make_candidate_examples,
    rank_candidates,
)
from src.utils.geo import batch_point_to_segment_projection, local_cartesian_projection


MODEL_FILES = {
    "KNN": "results/knn/pathsnap_knn.joblib",
    "Random Forest": "results/random_forest/pathsnap_random_forest.joblib",
}
METRIC_FILES = {
    "KNN": "results/knn/pathsnap_knn.metrics.json",
    "Random Forest": "results/random_forest/pathsnap_random_forest.metrics.json",
}
COLORS = {"KNN": "#3274a1", "Random Forest": "#e1812c"}


def _reference_arc_targets(trip, observation_ids: np.ndarray) -> np.ndarray:
    """Approximate each sampled GPS observation with its nearest route arc."""
    route_ids = np.unique(trip.route.to_numpy(dtype=np.int64))
    endpoints = trip.arcs.loc[route_ids, ["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    node_lon = trip.nodes.longitude.to_numpy(dtype=float)
    node_lat = trip.nodes.latitude.to_numpy(dtype=float)
    gps_lon = trip.track.longitude.to_numpy(dtype=float)
    gps_lat = trip.track.latitude.to_numpy(dtype=float)
    ref_lon = float(np.mean(np.concatenate((gps_lon, node_lon))))
    ref_lat = float(np.mean(np.concatenate((gps_lat, node_lat))))
    gps_x, gps_y, _, _ = local_cartesian_projection(gps_lon, gps_lat, ref_lat, ref_lon)
    node_x, node_y, _, _ = local_cartesian_projection(node_lon, node_lat, ref_lat, ref_lon)
    start = endpoints[:, 0]
    end = endpoints[:, 1]
    targets = []
    for observation_id in observation_ids:
        distances, _, _, _ = batch_point_to_segment_projection(
            float(gps_x[observation_id]),
            float(gps_y[observation_id]),
            node_x[start],
            node_y[start],
            node_x[end],
            node_y[end],
        )
        targets.append(int(route_ids[int(np.argmin(distances))]))
    return np.asarray(targets, dtype=np.int64)


def _route_coordinates(trip, route_arc_ids: list[int] | np.ndarray) -> list[list[float]]:
    """Convert ordered directed arc IDs to a Folium coordinate polyline."""
    points: list[list[float]] = []
    for arc_id in route_arc_ids:
        arc = trip.arcs.iloc[int(arc_id)]
        start = trip.nodes.iloc[int(arc.from_node_id)]
        end = trip.nodes.iloc[int(arc.to_node_id)]
        start_coord = [float(start.latitude), float(start.longitude)]
        end_coord = [float(end.latitude), float(end.longitude)]
        if not points or points[-1] != start_coord:
            points.append(start_coord)
        points.append(end_coord)
    return points


def _save_eda(data_root: Path, output_dir: Path, sample_trip_id: str) -> None:
    train_audit = audit_split(data_root / "train")
    student_audit = audit_split(data_root / "student_test")
    train_audit["split"] = "train"
    student_audit["split"] = "student_test"
    audit = pd.concat([train_audit, student_audit], ignore_index=True)
    audit.to_csv(output_dir / "trip_data_audit.csv", index=False)

    count_columns = ["track_rows", "node_rows", "arc_rows", "route_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, column in zip(axes.flat, count_columns):
        values = [audit.loc[audit.split == split, column].to_numpy() for split in ("train", "student_test")]
        ax.boxplot(values, tick_labels=["train", "student_test"], showfliers=False)
        ax.set_yscale("log")
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("Rows per trip (log scale)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Observed trip-file row counts by split (raw dataset)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_trip_row_counts.png", dpi=180)
    plt.close(fig)

    trip = load_trip(data_root / "student_test" / sample_trip_id, include_route=True)
    times = trip.track.timestamp_s.to_numpy(dtype=float)
    intervals = np.diff(times)
    lon = trip.track.longitude.to_numpy(dtype=float)
    lat = trip.track.latitude.to_numpy(dtype=float)
    from src.utils.geo import haversine_distance_vectorized

    distances = haversine_distance_vectorized(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speeds = distances / np.maximum(intervals, 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    scatter = axes[0].scatter(lon, lat, c=np.arange(len(lon)), s=9, cmap="viridis")
    axes[0].set_title(f"GPS trajectory: {sample_trip_id}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].axis("equal")
    fig.colorbar(scatter, ax=axes[0], label="Observation order")
    axes[1].hist(intervals, bins=30, color="#55a868", edgecolor="white")
    axes[1].set_title("GPS time deltas")
    axes[1].set_xlabel("Seconds between observations")
    axes[1].set_ylabel("Count")
    axes[2].hist(speeds[np.isfinite(speeds)], bins=40, color="#c44e52", edgecolor="white")
    axes[2].set_title("Observed step-speed distribution")
    axes[2].set_xlabel("m/s")
    axes[2].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_sample_trajectory.png", dpi=180)
    plt.close(fig)


def _plot_saved_comparisons(reports: dict[str, dict], output_dir: Path) -> None:
    models = list(reports)
    metric_keys = [
        ("mean_route_normalized_edit_similarity", "Route edit similarity"),
        ("mean_route_arc_f1", "Route arc F1"),
        ("mean_connected_transition_fraction", "Connected transitions"),
    ]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(13, 4.5))
    for ax, (key, title) in zip(axes, metric_keys):
        values = [reports[name]["heldout_test_route_metrics"][key] for name in models]
        bars = ax.bar(models, values, color=[COLORS[name] for name in models])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.3f", padding=2)
    fig.suptitle("Existing held-out route-sequence results (saved experiments)")
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_route_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in models:
        per_trip = pd.DataFrame(reports[name]["heldout_test_per_trip"])
        ax.plot(
            per_trip.trip_id,
            per_trip.route_arc_f1,
            marker="o",
            linewidth=1.5,
            label=name,
            color=COLORS[name],
        )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out trip ID")
    ax.set_ylabel("Route arc F1")
    ax.set_title("Per-trip held-out route F1 from saved reports")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_per_trip_route_f1.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    knn_tuning = pd.DataFrame(reports["KNN"]["validation_tuning_scores"])
    for weight in knn_tuning.weights.unique():
        weight_data = knn_tuning[knn_tuning.weights == weight].sort_values("n_neighbors")
        axes[0].plot(
            weight_data.n_neighbors,
            weight_data.proxy_top1_accuracy,
            marker="o",
            label=f"weights={weight}",
        )
    axes[0].set_title("KNN candidate tuning")
    axes[0].set_xlabel("n_neighbors")
    axes[0].set_ylabel("Validation proxy top-1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    forest_tuning = pd.DataFrame(reports["Random Forest"]["validation_candidate_tuning"])
    forest_labels = [
        f"trees={row.n_estimators}\nfeatures={row.max_features}\nleaf={row.min_samples_leaf}"
        for row in forest_tuning.itertuples()
    ]
    forest_bars = axes[1].bar(
        np.arange(len(forest_tuning)),
        forest_tuning.proxy_top1_accuracy,
        color=COLORS["Random Forest"],
    )
    axes[1].set_title("Random Forest candidate tuning")
    axes[1].set_ylabel("Validation proxy top-1")
    axes[1].set_xticks(np.arange(len(forest_tuning)), forest_labels, fontsize=8)
    axes[1].bar_label(forest_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    for name in models:
        key = "route_decoder_tuning_scores" if name == "KNN" else "validation_route_decoder_tuning"
        decoder = pd.DataFrame(reports[name][key]).sort_values("switch_penalty")
        axes[2].plot(
            decoder.switch_penalty,
            decoder.mean_route_normalized_edit_similarity,
            marker="o",
            label=name,
            color=COLORS[name],
        )
    axes[2].set_title("Route-decoder tuning")
    axes[2].set_xlabel("Switch penalty")
    axes[2].set_ylabel("Validation route similarity")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_tuning_results.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Cross-validation Macro F1: NOT AVAILABLE\n\nSaved experiments use one fixed trip-level fit/validation/holdout split.\nProducing CV scores requires refitting each model on CV folds.\nNo synthetic values or training curves are shown.",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f3f3f3", "edgecolor": "#777"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation_metric_availability.png", dpi=180)
    plt.close(fig)


def _evaluate_student_test(
    data_root: Path,
    output_dir: Path,
    models: dict[str, object],
    bundles: dict[str, dict],
    *,
    max_observations: int,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    from sklearn.metrics import f1_score

    rows = []
    predictions_for_map: dict[str, dict] = {}
    sample_trip_id = None
    for trip_dir in iter_trip_directories(data_root / "student_test"):
        trip = load_trip(trip_dir, include_route=True)
        sample_trip_id = sample_trip_id or trip.trip_id
        count = min(max_observations, len(trip.track))
        observation_ids = np.linspace(0, len(trip.track) - 1, count, dtype=int)
        targets = _reference_arc_targets(trip, observation_ids)
        for model_name, model in models.items():
            candidates = make_candidate_examples(
                trip,
                training=False,
                max_candidates=int(bundles[model_name]["max_candidates"]),
                max_observations=max_observations,
            )
            ranked = rank_candidates(model, candidates)
            # Candidate generation uses the same evenly spaced row positions.
            if not np.array_equal(ranked.observation_id.to_numpy(), observation_ids):
                raise RuntimeError(f"Observation sampling mismatch for {trip.trip_id}")
            predicted = ranked.arc_id.to_numpy(dtype=np.int64)
            accuracy = float(np.mean(predicted == targets))
            macro_f1 = float(
                f1_score(targets, predicted, labels=np.union1d(targets, predicted), average="macro", zero_division=0)
            )

            candidate_targets = dict(zip(observation_ids.tolist(), targets.tolist()))
            y_candidate = np.fromiter(
                (int(candidate_targets[int(row.observation_id)] == int(row.arc_id)) for row in candidates.itertuples()),
                dtype=np.int8,
                count=len(candidates),
            )
            y_candidate_pred = model.predict(candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            candidate_cm = confusion_matrix(y_candidate, y_candidate_pred, labels=[0, 1])
            probabilities = model.predict_proba(candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            class_ids = model.steps[-1][1].classes_
            positive_col = int(np.flatnonzero(class_ids == 1)[0])
            positive_probability = probabilities[:, positive_col]
            binary_probabilities = np.column_stack((1.0 - positive_probability, positive_probability))
            candidate_log_loss = float(log_loss(y_candidate, binary_probabilities, labels=[0, 1]))
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "model": model_name,
                    "sampled_observations": count,
                    "proxy_top1_accuracy": accuracy,
                    "proxy_macro_f1": macro_f1,
                    "candidate_proxy_log_loss": candidate_log_loss,
                    "candidate_tn": int(candidate_cm[0, 0]),
                    "candidate_fp": int(candidate_cm[0, 1]),
                    "candidate_fn": int(candidate_cm[1, 0]),
                    "candidate_tp": int(candidate_cm[1, 1]),
                }
            )
            if trip.trip_id == sample_trip_id:
                decoded = decode_connected_route(
                    model,
                    candidates,
                    trip.nodes,
                    trip.arcs,
                    switch_penalty=float(bundles[model_name].get("switch_penalty", 80.0)),
                )
                predictions_for_map[model_name] = {
                    "trip": trip,
                    "predicted_route": decoded,
                    "reference_route": trip.route.to_numpy(dtype=np.int64).tolist(),
                    "observation_ids": observation_ids,
                }
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "student_test_proxy_metrics_per_trip.csv", index=False)
    aggregate = frame.groupby("model").agg(
        proxy_top1_accuracy=("proxy_top1_accuracy", "mean"),
        proxy_macro_f1=("proxy_macro_f1", "mean"),
        candidate_proxy_log_loss=("candidate_proxy_log_loss", "mean"),
    )
    aggregate.to_csv(output_dir / "student_test_proxy_metrics_summary.csv")
    return aggregate, predictions_for_map


def _plot_student_comparison(aggregate: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    columns = ["proxy_top1_accuracy", "proxy_macro_f1"]
    titles = ["Top-1 accuracy vs route-derived proxy", "Macro F1 vs route-derived proxy"]
    models = aggregate.index.tolist()
    for ax, column, title in zip(axes, columns, titles):
        values = aggregate[column].to_numpy()
        bars = ax.bar(models, values, color=[COLORS[name] for name in models])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Mean across student_test trips")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Student-test inference: per-observation nearest-route-arc proxy labels")
    fig.tight_layout()
    fig.savefig(output_dir / "student_test_top1_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(10, 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        data = pd.read_csv(output_dir / "student_test_proxy_metrics_per_trip.csv")
        data = data[data.model == name]
        cm = np.array(
            [
                [data.candidate_tn.sum(), data.candidate_fp.sum()],
                [data.candidate_fn.sum(), data.candidate_tp.sum()],
            ],
            dtype=int,
        )
        ConfusionMatrixDisplay(cm, display_labels=["other arc", "nearest route arc"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False
        )
        ax.set_title(f"{name}: candidate proxy labels")
    fig.suptitle("Optional binary candidate confusion matrices (proxy labels)")
    fig.tight_layout()
    fig.savefig(output_dir / "student_test_candidate_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = aggregate["candidate_proxy_log_loss"].reindex(models).to_numpy()
    bars = ax.bar(models, values, color=[COLORS[name] for name in models])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Binary candidate log loss (lower is better)")
    ax.set_title("Student-test candidate log loss vs route-derived proxy labels")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "student_test_candidate_log_loss.png", dpi=180)
    plt.close(fig)


def _save_folium_map(predictions: dict[str, dict], output_dir: Path) -> str:
    model_names = list(predictions)
    first = predictions[model_names[0]]
    trip = first["trip"]
    center = [float(trip.track.latitude.mean()), float(trip.track.longitude.mean())]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    gps = list(zip(trip.track.latitude.astype(float), trip.track.longitude.astype(float)))
    folium.PolyLine(gps, color="#555555", weight=3, opacity=0.8, tooltip="Observed GPS track").add_to(
        folium.FeatureGroup(name="GPS observations", show=True).add_to(fmap)
    )
    ref_group = folium.FeatureGroup(name="Reference .route", show=True)
    reference_coords = _route_coordinates(trip, first["reference_route"])
    folium.PolyLine(reference_coords, color="#2ca02c", weight=5, opacity=0.8, tooltip="Reference route").add_to(ref_group)
    ref_group.add_to(fmap)

    for model_name in model_names:
        entry = predictions[model_name]
        group = folium.FeatureGroup(name=f"{model_name} connected prediction", show=(model_name == "Random Forest"))
        coords = _route_coordinates(trip, entry["predicted_route"])
        folium.PolyLine(
            coords,
            color=COLORS[model_name],
            weight=4,
            opacity=0.75,
            tooltip=f"{model_name} connected predicted route ({len(entry['predicted_route'])} arcs)",
        ).add_to(group)
        group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    folium.map.Marker(center, tooltip=f"student_test trip {trip.trip_id}").add_to(fmap)
    path = output_dir / f"folium_student_test_trip_{trip.trip_id}.html"
    fmap.save(path)
    return str(path)


def generate_visualizations(
    project_root: str | Path = ".",
    *,
    sample_trip_id: str = "00000000",
    max_test_observations: int = 500,
) -> Path:
    """Generate figures and maps from saved artifacts, without fitting models."""
    root = Path(project_root).resolve()
    output_dir = root / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        name: json.loads((root / METRIC_FILES[name]).read_text()) for name in MODEL_FILES
    }

    _save_eda(root / "data" / "Student folder", output_dir, sample_trip_id)
    _plot_saved_comparisons(reports, output_dir)

    import joblib

    bundles = {name: joblib.load(root / path) for name, path in MODEL_FILES.items()}
    models = {name: bundles[name]["model"] for name in MODEL_FILES}
    student_aggregate, predictions = _evaluate_student_test(
        root / "data" / "Student folder",
        output_dir,
        models,
        bundles,
        max_observations=max_test_observations,
    )
    _plot_student_comparison(student_aggregate, output_dir)
    map_path = _save_folium_map(predictions, output_dir)

    availability = {
        "cross_validation_macro_f1": "not_generated: would require new fold-specific model fits, which were explicitly not requested",
        "student_test_top1_and_macro_f1": "generated against a route-derived nearest-arc proxy; .route does not align labels one-to-one with GPS observations",
        "training_validation_loss_curves": "not_applicable: KNN and Random Forest do not produce epoch-wise loss histories",
        "student_test_metrics": student_aggregate.to_dict(orient="index"),
        "folium_map": map_path,
        "static_figures": sorted(path.name for path in output_dir.glob("*.png")),
        "uses_model_retraining": False,
    }
    (output_dir / "visualization_report.json").write_text(json.dumps(availability, indent=2))
    (output_dir / "README.md").write_text(
        "# Model visualizations\n\n"
        "Generated by loading saved estimators and reports; no model fitting is performed.\n\n"
        "Student-test top-1 and macro-F1 are against a proxy target: for each sampled GPS point, the nearest arc in its supplied reference route. They are not direct per-observation ground truth. The held-out route comparison plots use previously saved route-sequence evaluation results.\n\n"
        "Cross-validation Macro F1 is not present in the saved experiments. Producing it properly requires fitting each model in multiple CV folds; this was skipped to honor the no-retraining request. Neither KNN nor Random Forest has epoch-based training histories, so loss curves are not applicable.\n\n"
        "See `visualization_report.json` for generated outputs and exact aggregate student-test proxy metrics.\n"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--sample-trip-id", default="00000000")
    parser.add_argument("--max-test-observations", type=int, default=500)
    args = parser.parse_args()
    output_dir = generate_visualizations(
        args.project_root,
        sample_trip_id=args.sample_trip_id,
        max_test_observations=args.max_test_observations,
    )
    print(f"Visualization outputs saved under {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
