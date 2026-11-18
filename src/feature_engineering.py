"""Feature extraction for GPS observations and road-network segments.

GPS-derived rows and road-segment rows stay separate: route labels are arc IDs
and do not have a one-to-one relationship with GPS observations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Also allow ``python src/feature_engineering.py`` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import TripData, iter_trip_directories, load_trip
from src.utils.geo import haversine_distance_vectorized, initial_bearing_vectorized


def trajectory_features(track: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per GPS observation in chronological order."""
    required = {"longitude", "latitude", "timestamp_s"}
    missing = required - set(track.columns)
    if missing:
        raise ValueError(f"Track is missing columns: {sorted(missing)}")
    if track.empty:
        return pd.DataFrame(
            columns=[
                "longitude", "latitude", "elapsed_time_s", "delta_time_s",
                "step_distance_m", "speed_mps", "acceleration_mps2",
                "heading_sin", "heading_cos", "heading_valid",
            ],
            index=pd.Index([], name="observation_id"),
        )

    values = track[["longitude", "latitude", "timestamp_s"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Track features require finite coordinates and timestamps")

    longitude = values[:, 0]
    latitude = values[:, 1]
    timestamp = values[:, 2]
    delta_time = np.r_[0.0, np.diff(timestamp)]
    if np.any(delta_time[1:] <= 0):
        raise ValueError("Track timestamps must be strictly increasing; run clean_track first")

    distance = np.zeros(len(track), dtype=float)
    bearing = np.zeros(len(track), dtype=float)
    if len(track) > 1:
        distance[1:] = haversine_distance_vectorized(
            longitude[:-1], latitude[:-1], longitude[1:], latitude[1:]
        )
        bearing[1:] = initial_bearing_vectorized(
            longitude[:-1], latitude[:-1], longitude[1:], latitude[1:]
        )

    speed = np.divide(distance, delta_time, out=np.zeros_like(distance), where=delta_time > 0)
    acceleration = np.zeros(len(track), dtype=float)
    if len(track) > 1:
        acceleration[1:] = np.diff(speed) / delta_time[1:]
    bearing_radians = np.radians(bearing)
    heading_valid = distance > 1e-6

    features = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "elapsed_time_s": timestamp - timestamp[0],
            "delta_time_s": delta_time,
            "step_distance_m": distance,
            "speed_mps": speed,
            "acceleration_mps2": acceleration,
            "heading_sin": np.where(heading_valid, np.sin(bearing_radians), 0.0),
            "heading_cos": np.where(heading_valid, np.cos(bearing_radians), 0.0),
            "heading_valid": heading_valid.astype(np.int8),
        },
        index=pd.RangeIndex(len(track), name="observation_id"),
    )
    return features


def road_segment_features(nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per directed arc, retaining the original arc ID."""
    required_nodes = {"longitude", "latitude"}
    required_arcs = {"from_node_id", "to_node_id"}
    if required_nodes - set(nodes.columns) or required_arcs - set(arcs.columns):
        raise ValueError("Nodes/arcs do not have the expected named columns")
    if arcs.empty:
        return pd.DataFrame(index=pd.Index([], name="arc_id"))

    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    if endpoints.min() < 0 or endpoints.max() >= len(nodes):
        raise ValueError("Arc endpoint references a missing node")

    coordinates = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Road-network nodes must have finite coordinates")
    from_ids, to_ids = endpoints.T
    start = coordinates[from_ids]
    end = coordinates[to_ids]
    length = haversine_distance_vectorized(start[:, 0], start[:, 1], end[:, 0], end[:, 1])
    bearing = initial_bearing_vectorized(start[:, 0], start[:, 1], end[:, 0], end[:, 1])
    bearing_radians = np.radians(bearing)

    out_degree = np.bincount(from_ids, minlength=len(nodes))
    in_degree = np.bincount(to_ids, minlength=len(nodes))
    features = pd.DataFrame(
        {
            "from_node_id": from_ids,
            "to_node_id": to_ids,
            "from_longitude": start[:, 0],
            "from_latitude": start[:, 1],
            "to_longitude": end[:, 0],
            "to_latitude": end[:, 1],
            "midpoint_longitude": (start[:, 0] + end[:, 0]) / 2.0,
            "midpoint_latitude": (start[:, 1] + end[:, 1]) / 2.0,
            "length_m": length,
            "bearing_sin": np.sin(bearing_radians),
            "bearing_cos": np.cos(bearing_radians),
            "from_out_degree": out_degree[from_ids],
            "to_in_degree": in_degree[to_ids],
        },
        index=pd.Index(arcs.index.to_numpy(), name="arc_id"),
    )
    return features


def prepare_trip_features(trip: TripData) -> dict[str, object]:
    """Prepare model-ready tables without falsely aligning GPS rows and arcs.

    ``route_arc_ids`` is returned separately as a sequence of target arc IDs
    when the caller explicitly loaded labels. It should not be used as an
    input feature.
    """
    return {
        "trip_id": trip.trip_id,
        "trajectory": trajectory_features(trip.track),
        "road_segments": road_segment_features(trip.nodes, trip.arcs),
        "route_arc_ids": None if trip.route is None else trip.route.to_numpy(copy=True),
    }


def export_features(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    trip_id: str | None = None,
) -> pd.DataFrame:
    """Export per-trip feature tables and training routes as CSV files.

    The route is loaded only for the training split, and saved separately from
    the feature tables. For a smaller smoke test, pass a single ``trip_id``.
    """
    if split not in {"train", "student_test"}:
        raise ValueError("split must be 'train' or 'student_test'")

    split_dir = Path(data_root) / split
    trip_dirs = list(iter_trip_directories(split_dir))
    if trip_id is not None:
        trip_dirs = [path for path in trip_dirs if path.name == trip_id]
        if not trip_dirs:
            raise FileNotFoundError(f"Trip {trip_id!r} was not found in {split_dir}")

    output_root = Path(output_dir) / split
    summary = []
    for trip_dir in trip_dirs:
        trip = load_trip(trip_dir, include_route=(split == "train"))
        features = prepare_trip_features(trip)
        trip_output = output_root / trip.trip_id
        trip_output.mkdir(parents=True, exist_ok=True)

        features["trajectory"].to_csv(trip_output / "trajectory_features.csv", index_label="observation_id")
        features["road_segments"].to_csv(trip_output / "road_segment_features.csv", index_label="arc_id")
        route = features["route_arc_ids"]
        if route is not None:
            pd.DataFrame(
                {"route_position": np.arange(len(route)), "arc_id": route}
            ).to_csv(trip_output / "route_labels.csv", index=False)

        summary.append(
            {
                "split": split,
                "trip_id": trip.trip_id,
                "gps_observations": len(features["trajectory"]),
                "road_segments": len(features["road_segments"]),
                "route_labels": 0 if route is None else len(route),
            }
        )

    report = pd.DataFrame(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_root / "summary.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and export per-trip GPS and road-network features."
    )
    parser.add_argument(
        "--data-root",
        default="data/Student folder",
        help="Dataset root containing train/ and student_test/ directories.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "student_test"),
        help="Which dataset split to process. Route labels are exported only for train.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/features",
        help="Directory for the per-trip CSV exports (default: results/features).",
    )
    parser.add_argument(
        "--trip-id",
        help="Optionally process only one trip, e.g. 00000001.",
    )
    args = parser.parse_args()

    report = export_features(
        args.data_root,
        args.output_dir,
        split=args.split,
        trip_id=args.trip_id,
    )
    print(f"Exported {len(report)} trip(s) to {Path(args.output_dir) / args.split}")
    if not report.empty:
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()