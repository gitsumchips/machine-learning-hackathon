"""Build route-labeled road candidates and train a binary Logistic Regression."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_processing import iter_trip_directories
from src.utils.geo import (
    batch_point_to_segment_projection,
    local_cartesian_projection,
)

CANDIDATE_FEATURES = [
    "perpendicular_distance_m",
    "along_track_distance_m",
    "segment_length_m",
    "gps_step_distance_m",
    "gps_speed_mps",
    "gps_acceleration_mps2",
    "gps_delta_time_s",
    "heading_alignment",
    "heading_valid",
    "arc_bearing_sin",
    "arc_bearing_cos",
    "from_out_degree",
    "to_in_degree",
]


def _segment_grid_cells(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    cell_size_m: float,
) -> list[tuple[int, int]]:
    """Return grid cells touched by a segment using 2D voxel traversal."""
    cell_x = math.floor(start_x / cell_size_m)
    cell_y = math.floor(start_y / cell_size_m)
    end_cell_x = math.floor(end_x / cell_size_m)
    end_cell_y = math.floor(end_y / cell_size_m)
    cells = [(cell_x, cell_y)]

    delta_x = end_x - start_x
    delta_y = end_y - start_y
    step_x = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
    step_y = 1 if delta_y > 0 else -1 if delta_y < 0 else 0

    if step_x:
        boundary_x = (cell_x + (step_x > 0)) * cell_size_m
        max_t_x = (boundary_x - start_x) / delta_x
        delta_t_x = cell_size_m / abs(delta_x)
    else:
        max_t_x = delta_t_x = math.inf

    if step_y:
        boundary_y = (cell_y + (step_y > 0)) * cell_size_m
        max_t_y = (boundary_y - start_y) / delta_y
        delta_t_y = cell_size_m / abs(delta_y)
    else:
        max_t_y = delta_t_y = math.inf

    while (cell_x, cell_y) != (end_cell_x, end_cell_y):
        if max_t_x < max_t_y:
            cell_x += step_x
            max_t_x += delta_t_x
        elif max_t_y < max_t_x:
            cell_y += step_y
            max_t_y += delta_t_y
        else:
            if step_x:
                cells.append((cell_x + step_x, cell_y))
            if step_y:
                cells.append((cell_x, cell_y + step_y))
            cell_x += step_x
            cell_y += step_y
            max_t_x += delta_t_x
            max_t_y += delta_t_y
        cells.append((cell_x, cell_y))

    return cells


class SegmentGridIndex:
    """Spatial index for nearby directed road segments in local metric coordinates."""

    def __init__(
        self,
        arc_ids: np.ndarray,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray,
        cell_size_m: float,
    ) -> None:
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        if len(set(arc_ids.tolist())) != len(arc_ids):
            raise ValueError("arc_ids must be unique within a trip")

        self.arc_ids = np.asarray(arc_ids, dtype=np.int64)
        self.start_x = np.asarray(start_x, dtype=float)
        self.start_y = np.asarray(start_y, dtype=float)
        self.end_x = np.asarray(end_x, dtype=float)
        self.end_y = np.asarray(end_y, dtype=float)
        self.cell_size_m = float(cell_size_m)
        self.segment_lengths = np.hypot(
            self.end_x - self.start_x, self.end_y - self.start_y
        )
        self.cells: dict[tuple[int, int], list[int]] = {}

        for row_idx in range(len(self.arc_ids)):
            cells = _segment_grid_cells(
                self.start_x[row_idx],
                self.start_y[row_idx],
                self.end_x[row_idx],
                self.end_y[row_idx],
                self.cell_size_m,
            )
            for cell in cells:
                self.cells.setdefault(cell, []).append(row_idx)

    def query(
        self,
        x: float,
        y: float,
        *,
        top_k: int,
        max_distance_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arc IDs, row indices, distances, and projection fractions."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")

        center_x = math.floor(x / self.cell_size_m)
        center_y = math.floor(y / self.cell_size_m)
        window = math.ceil(max_distance_m / self.cell_size_m) + 1
        nearby_rows: set[int] = set()
        for offset_x in range(-window, window + 1):
            for offset_y in range(-window, window + 1):
                nearby_rows.update(
                    self.cells.get((center_x + offset_x, center_y + offset_y), ())
                )

        if not nearby_rows:
            empty_int = np.empty(0, dtype=np.int64)
            empty_float = np.empty(0, dtype=float)
            return empty_int, empty_int, empty_float, empty_float

        row_indices = np.fromiter(sorted(nearby_rows), dtype=np.int64)
        distances, projections, _, _ = batch_point_to_segment_projection(
            x,
            y,
            self.start_x[row_indices],
            self.start_y[row_indices],
            self.end_x[row_indices],
            self.end_y[row_indices],
        )
        within_radius = distances <= max_distance_m
        row_indices = row_indices[within_radius]
        distances = distances[within_radius]
        projections = projections[within_radius]
        order = np.lexsort((self.arc_ids[row_indices], distances))[:top_k]
        row_indices = row_indices[order]
        return (
            self.arc_ids[row_indices],
            row_indices,
            distances[order],
            projections[order],
        )


def align_route_to_track(
    track_x: np.ndarray,
    track_y: np.ndarray,
    route_arc_ids: np.ndarray,
    arc_ids: np.ndarray,
    arc_start_x: np.ndarray,
    arc_start_y: np.ndarray,
    arc_end_x: np.ndarray,
    arc_end_y: np.ndarray,
    *,
    track_elapsed_time_s: np.ndarray | None = None,
    max_route_speed_mps: float = 80.0,
    start_tolerance_m: float = 200.0,
    end_tolerance_m: float = 600.0,
    advance_penalty_m: float = 5.0,
) -> np.ndarray:
    """Align GPS observations in route order with a time-based travel limit."""
    if advance_penalty_m < 0:
        raise ValueError("advance_penalty_m cannot be negative")
    if max_route_speed_mps <= 0:
        raise ValueError("max_route_speed_mps must be positive")
    if start_tolerance_m < 0 or end_tolerance_m < 0:
        raise ValueError("route endpoint tolerances cannot be negative")
    if len(track_x) != len(track_y):
        raise ValueError("track_x and track_y must have equal lengths")
    if len(track_x) == 0 or len(route_arc_ids) == 0:
        raise ValueError("track and route must both contain at least one row")
    if track_elapsed_time_s is None:
        elapsed_time_s = np.arange(len(track_x), dtype=float)
    else:
        elapsed_time_s = np.asarray(track_elapsed_time_s, dtype=float)
        if len(elapsed_time_s) != len(track_x) or not np.isfinite(elapsed_time_s).all():
            raise ValueError("track_elapsed_time_s must be finite and match the track")
        if np.any(np.diff(elapsed_time_s) <= 0):
            raise ValueError("track_elapsed_time_s must be strictly increasing")

    row_by_arc_id = {int(arc_id): row for row, arc_id in enumerate(arc_ids)}
    try:
        route_rows = np.fromiter(
            (row_by_arc_id[int(arc_id)] for arc_id in route_arc_ids),
            dtype=np.int64,
            count=len(route_arc_ids),
        )
    except KeyError as exc:
        raise ValueError(f"Route references unknown arc ID {exc.args[0]}") from exc

    route_start_x = arc_start_x[route_rows]
    route_start_y = arc_start_y[route_rows]
    route_end_x = arc_end_x[route_rows]
    route_end_y = arc_end_y[route_rows]
    num_observations = len(track_x)
    num_route_positions = len(route_arc_ids)

    emissions = np.empty((num_observations, num_route_positions), dtype=np.float32)
    projections = np.empty_like(emissions)
    for obs_idx, (x, y) in enumerate(zip(track_x, track_y)):
        distances, fractions, _, _ = batch_point_to_segment_projection(
            float(x),
            float(y),
            route_start_x,
            route_start_y,
            route_end_x,
            route_end_y,
        )
        emissions[obs_idx] = distances
        projections[obs_idx] = fractions

    positions = np.arange(num_route_positions, dtype=np.float32)
    route_lengths = np.hypot(
        route_end_x - route_start_x, route_end_y - route_start_y
    )
    route_distance = np.r_[0.0, np.cumsum(route_lengths[:-1])]
    backpointers = np.empty(
        (num_observations, num_route_positions), dtype=np.int32
    )
    start_positions = route_distance <= start_tolerance_m
    costs = np.where(start_positions, emissions[0], np.inf)

    for obs_idx in range(1, num_observations):
        adjusted_costs = costs - advance_penalty_m * positions
        max_advance_m = max_route_speed_mps * (
            elapsed_time_s[obs_idx] - elapsed_time_s[obs_idx - 1]
        )
        previous_progress = (
            route_distance
            + projections[obs_idx - 1] * route_lengths
        )
        current_progress = (
            route_distance + projections[obs_idx] * route_lengths
        )
        earliest_position = np.searchsorted(
            previous_progress,
            current_progress - max_advance_m,
            side="left",
        )
        earliest_position = np.minimum(
            earliest_position, np.arange(num_route_positions)
        )
        range_lengths = np.arange(num_route_positions) - earliest_position + 1
        range_levels = np.floor(np.log2(range_lengths)).astype(np.int64)

        minimum_values = [adjusted_costs]
        minimum_indices = [np.arange(num_route_positions, dtype=np.int32)]
        level = 1
        while (1 << level) <= num_route_positions:
            half_span = 1 << (level - 1)
            valid_count = num_route_positions - (1 << level) + 1
            previous_values = minimum_values[-1]
            previous_indices = minimum_indices[-1]
            left_values = previous_values[:valid_count]
            right_values = previous_values[
                half_span : half_span + valid_count
            ]
            left_indices = previous_indices[:valid_count]
            right_indices = previous_indices[
                half_span : half_span + valid_count
            ]
            choose_left = (left_values < right_values) | (
                (left_values == right_values) & (left_indices < right_indices)
            )
            current_values = np.full(
                num_route_positions, np.inf, dtype=adjusted_costs.dtype
            )
            current_indices = np.full(
                num_route_positions, -1, dtype=np.int32
            )
            current_values[:valid_count] = np.where(
                choose_left, left_values, right_values
            )
            current_indices[:valid_count] = np.where(
                choose_left, left_indices, right_indices
            )
            minimum_values.append(current_values)
            minimum_indices.append(current_indices)
            level += 1

        predecessor = np.empty(num_route_positions, dtype=np.int32)
        for level in range(int(range_levels.max()) + 1):
            mask = range_levels == level
            if not np.any(mask):
                continue
            span = 1 << level
            start_positions = earliest_position[mask]
            end_positions = np.flatnonzero(mask)
            left_values = minimum_values[level][start_positions]
            right_positions = end_positions - span + 1
            right_values = minimum_values[level][right_positions]
            left_indices = minimum_indices[level][start_positions]
            right_indices = minimum_indices[level][right_positions]
            choose_left = (left_values < right_values) | (
                (left_values == right_values) & (left_indices < right_indices)
            )
            predecessor[mask] = np.where(
                choose_left, left_indices, right_indices
            )
        backpointers[obs_idx] = predecessor
        costs = (
            emissions[obs_idx]
            + advance_penalty_m * positions
            + adjusted_costs[predecessor]
        )

    end_positions = route_distance >= route_distance[-1] - end_tolerance_m
    costs[~end_positions] = np.inf
    if not np.isfinite(costs).any():
        reachable_positions = np.flatnonzero(np.isfinite(costs))
        furthest_position = (
            int(reachable_positions[-1]) if len(reachable_positions) else -1
        )
        remaining_route_distance_m = (
            float(route_distance[-1] - route_distance[furthest_position])
            if furthest_position >= 0
            else float(route_distance[-1])
        )
        raise ValueError(
            "Track cannot reach the end of the reference route within the "
            "configured speed and endpoint tolerances; furthest reachable "
            f"route position={furthest_position}/{num_route_positions - 1}, "
            f"remaining route distance={remaining_route_distance_m:.1f} m"
        )

    route_positions = np.empty(num_observations, dtype=np.int64)
    route_positions[-1] = int(np.argmin(costs))
    for obs_idx in range(num_observations - 1, 0, -1):
        route_positions[obs_idx - 1] = backpointers[
            obs_idx, route_positions[obs_idx]
        ]
    return route_arc_ids[route_positions]


def _read_trip_feature_tables(
    trip_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    trajectory_path = trip_dir / "trajectory_features.csv"
    roads_path = trip_dir / "road_segment_features.csv"
    route_path = trip_dir / "route_labels.csv"
    for path in (trajectory_path, roads_path, route_path):
        if not path.is_file():
            raise FileNotFoundError(f"Training feature file is missing: {path}")

    trajectory = pd.read_csv(trajectory_path)
    roads = pd.read_csv(roads_path)
    route = pd.read_csv(route_path)
    required_trajectory = {
        "observation_id",
        "longitude",
        "latitude",
        "elapsed_time_s",
        "delta_time_s",
        "step_distance_m",
        "speed_mps",
        "acceleration_mps2",
        "heading_sin",
        "heading_cos",
        "heading_valid",
    }
    required_roads = {
        "arc_id",
        "from_node_id",
        "to_node_id",
        "from_longitude",
        "from_latitude",
        "to_longitude",
        "to_latitude",
        "bearing_sin",
        "bearing_cos",
        "from_out_degree",
        "to_in_degree",
    }
    if missing := required_trajectory - set(trajectory.columns):
        raise ValueError(f"{trajectory_path} is missing columns: {sorted(missing)}")
    if missing := required_roads - set(roads.columns):
        raise ValueError(f"{roads_path} is missing columns: {sorted(missing)}")
    if "arc_id" not in route.columns:
        raise ValueError(f"{route_path} is missing the arc_id column")
    if trajectory.empty or roads.empty or route.empty:
        raise ValueError(f"Trip {trip_dir.name} has an empty track, network, or route")
    trajectory = trajectory.sort_values("observation_id", kind="stable").reset_index(
        drop=True
    )
    if not np.array_equal(
        trajectory["observation_id"].to_numpy(), np.arange(len(trajectory))
    ):
        raise ValueError(
            f"Trip {trip_dir.name} observation IDs must be unique and zero-based"
        )
    elapsed_time_s = trajectory["elapsed_time_s"].to_numpy(dtype=float)
    delta_time_s = trajectory["delta_time_s"].to_numpy(dtype=float)
    if not np.isfinite(elapsed_time_s).all() or not np.isfinite(delta_time_s).all():
        raise ValueError(f"Trip {trip_dir.name} has invalid observation timestamps")
    if (
        (len(trajectory) > 1 and np.any(np.diff(elapsed_time_s) <= 0))
        or not np.isclose(delta_time_s[0], 0.0)
        or not np.allclose(
            delta_time_s[1:], np.diff(elapsed_time_s), rtol=0.0, atol=1e-6
        )
    ):
        raise ValueError(
            f"Trip {trip_dir.name} observation timestamps are not chronological"
        )

    road_arc_ids = roads["arc_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(road_arc_ids, np.arange(len(roads))):
        raise ValueError(
            f"Trip {trip_dir.name} arc IDs must match their zero-based row indices"
        )

    if "route_position" in route.columns:
        route = route.sort_values("route_position", kind="stable")
        route_positions = route["route_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(route_positions, np.arange(len(route))):
            raise ValueError(
                f"Trip {trip_dir.name} route positions must be unique and contiguous"
            )
    if not route["arc_id"].isin(roads["arc_id"]).all():
        raise ValueError(f"Trip {trip_dir.name} route references an unknown arc ID")
    return trajectory, roads, route["arc_id"].to_numpy(dtype=np.int64)


def generate_trip_candidates(
    trip_dir: str | Path,
    *,
    top_k: int = 128,
    max_distance_m: float = 200.0,
    cell_size_m: float = 100.0,
    max_route_speed_mps: float = 80.0,
    start_tolerance_m: float = 200.0,
    end_tolerance_m: float = 600.0,
    advance_penalty_m: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate top-K road candidates and binary reference labels for one training trip."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_distance_m <= 0 or cell_size_m <= 0:
        raise ValueError("max_distance_m and cell_size_m must be positive")
    if max_route_speed_mps <= 0 or start_tolerance_m < 0 or end_tolerance_m < 0:
        raise ValueError("route speed must be positive and endpoint tolerances nonnegative")
    trip_path = Path(trip_dir)
    trajectory, roads, route_arc_ids = _read_trip_feature_tables(trip_path)

    track_values = trajectory[
        ["longitude", "latitude", "delta_time_s", "step_distance_m", "speed_mps",
         "acceleration_mps2", "heading_sin", "heading_cos", "heading_valid"]
    ].to_numpy(dtype=float)
    road_values = roads[
        ["from_longitude", "from_latitude", "to_longitude", "to_latitude",
         "bearing_sin", "bearing_cos", "from_out_degree", "to_in_degree"]
    ].to_numpy(dtype=float)
    if not np.isfinite(track_values).all() or not np.isfinite(road_values).all():
        raise ValueError(f"Trip {trip_path.name} contains non-finite features")
    if (
        not trajectory["longitude"].between(-180, 180).all()
        or not trajectory["latitude"].between(-90, 90).all()
        or not roads["from_longitude"].between(-180, 180).all()
        or not roads["to_longitude"].between(-180, 180).all()
        or not roads["from_latitude"].between(-90, 90).all()
        or not roads["to_latitude"].between(-90, 90).all()
    ):
        raise ValueError(f"Trip {trip_path.name} contains out-of-range coordinates")

    reference_lon = float(trajectory["longitude"].mean())
    reference_lat = float(trajectory["latitude"].mean())
    track_x, track_y, _, _ = local_cartesian_projection(
        trajectory["longitude"].to_numpy(),
        trajectory["latitude"].to_numpy(),
        ref_lat=reference_lat,
        ref_lon=reference_lon,
    )
    start_x, start_y, _, _ = local_cartesian_projection(
        roads["from_longitude"].to_numpy(),
        roads["from_latitude"].to_numpy(),
        ref_lat=reference_lat,
        ref_lon=reference_lon,
    )
    end_x, end_y, _, _ = local_cartesian_projection(
        roads["to_longitude"].to_numpy(),
        roads["to_latitude"].to_numpy(),
        ref_lat=reference_lat,
        ref_lon=reference_lon,
    )
    arc_ids = roads["arc_id"].to_numpy(dtype=np.int64)
    row_by_arc_id = {int(arc_id): row for row, arc_id in enumerate(arc_ids)}
    route_rows = np.fromiter(
        (row_by_arc_id[int(arc_id)] for arc_id in route_arc_ids),
        dtype=np.int64,
        count=len(route_arc_ids),
    )
    route_segments = roads.iloc[route_rows]
    route_node_disconnects = int(
        np.count_nonzero(
            route_segments["to_node_id"].to_numpy()[:-1]
            != route_segments["from_node_id"].to_numpy()[1:]
        )
    )
    route_geometry_gaps = np.hypot(
        start_x[route_rows][1:] - end_x[route_rows][:-1],
        start_y[route_rows][1:] - end_y[route_rows][:-1],
    )
    assigned_arc_ids = align_route_to_track(
        track_x,
        track_y,
        route_arc_ids,
        arc_ids,
        start_x,
        start_y,
        end_x,
        end_y,
        track_elapsed_time_s=trajectory["elapsed_time_s"].to_numpy(dtype=float),
        max_route_speed_mps=max_route_speed_mps,
        start_tolerance_m=start_tolerance_m,
        end_tolerance_m=end_tolerance_m,
        advance_penalty_m=advance_penalty_m,
    )
    index = SegmentGridIndex(
        arc_ids,
        start_x,
        start_y,
        end_x,
        end_y,
        cell_size_m,
    )

    capacity = len(trajectory) * top_k
    feature_values = np.empty((capacity, len(CANDIDATE_FEATURES)), dtype=np.float32)
    candidate_trip_ids: list[str] = []
    candidate_observation_ids = np.empty(capacity, dtype=np.int64)
    candidate_arc_ids = np.empty(capacity, dtype=np.int64)
    candidate_labels = np.empty(capacity, dtype=np.int8)
    cursor = 0
    candidate_hits = 0
    zero_candidate_observations = 0
    candidate_observations = 0
    candidates_before_label_filter = 0

    for obs_idx, row in enumerate(trajectory.itertuples(index=False)):
        arc_ids_for_obs, row_indices, distances, projections = index.query(
            float(track_x[obs_idx]),
            float(track_y[obs_idx]),
            top_k=top_k,
            max_distance_m=max_distance_m,
        )
        if not len(arc_ids_for_obs):
            zero_candidate_observations += 1
            continue

        candidate_observations += 1
        candidates_before_label_filter += len(arc_ids_for_obs)
        truth_arc_id = assigned_arc_ids[obs_idx]
        labels = (arc_ids_for_obs == truth_arc_id).astype(np.int8)
        if not np.any(labels):
            continue
        candidate_hits += 1
        segment_lengths = index.segment_lengths[row_indices]
        heading_alignment = (
            row.heading_sin * roads["bearing_sin"].to_numpy()[row_indices]
            + row.heading_cos * roads["bearing_cos"].to_numpy()[row_indices]
        )
        count = len(arc_ids_for_obs)
        stop = cursor + count
        feature_values[cursor:stop] = np.column_stack(
            (
                distances,
                projections * segment_lengths,
                segment_lengths,
                np.full(count, row.step_distance_m),
                np.full(count, row.speed_mps),
                np.full(count, row.acceleration_mps2),
                np.full(count, row.delta_time_s),
                heading_alignment,
                np.full(count, row.heading_valid),
                roads["bearing_sin"].to_numpy()[row_indices],
                roads["bearing_cos"].to_numpy()[row_indices],
                roads["from_out_degree"].to_numpy()[row_indices],
                roads["to_in_degree"].to_numpy()[row_indices],
            )
        )
        candidate_observation_ids[cursor:stop] = row.observation_id
        candidate_arc_ids[cursor:stop] = arc_ids_for_obs
        candidate_labels[cursor:stop] = labels
        candidate_trip_ids.extend([trip_path.name] * count)
        cursor = stop

    candidates = pd.DataFrame(feature_values[:cursor], columns=CANDIDATE_FEATURES)
    candidates.insert(0, "arc_id", candidate_arc_ids[:cursor])
    candidates.insert(0, "observation_id", candidate_observation_ids[:cursor])
    candidates.insert(0, "trip_id", candidate_trip_ids)
    candidates["label"] = candidate_labels[:cursor]

    total_observations = len(trajectory)
    summary = {
        "trip_id": trip_path.name,
        "observations": total_observations,
        "candidates": cursor,
        "candidates_before_label_filter": candidates_before_label_filter,
        "positive_candidates": int(candidate_labels[:cursor].sum()),
        "negative_candidates": int(cursor - candidate_labels[:cursor].sum()),
        "candidate_observations": candidate_observations,
        "labeled_observations": candidate_hits,
        "excluded_unmatched_observations": total_observations - candidate_hits,
        "zero_candidate_observations": zero_candidate_observations,
        "route_node_disconnect_transitions": route_node_disconnects,
        "route_max_geometry_gap_m": float(route_geometry_gaps.max(initial=0.0)),
        "candidate_hit_observations": candidate_hits,
        "candidate_recall_pct": 100.0 * candidate_hits / total_observations,
    }
    return candidates, summary


def train_logistic_regression(
    candidates: pd.DataFrame,
    *,
    model_path: str | Path,
    validation_size: float = 0.2,
    random_seed: int = 42,
    candidate_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate by held-out trips, refit on all training trips, and save the model."""
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    required = {"trip_id", "observation_id", "arc_id", "label", *CANDIDATE_FEATURES}
    if missing := required - set(candidates.columns):
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if candidates.empty or candidates["label"].isna().any():
        raise ValueError("Training candidates must be non-empty and fully labeled")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    groups = candidates["trip_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two trips are required for grouped validation")
    features = candidates[CANDIDATE_FEATURES].to_numpy(dtype=np.float32)
    labels = candidates["label"].to_numpy(dtype=np.int8)
    if not np.isfinite(features).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Training features must be finite and labels must be binary")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=random_seed
    )
    train_indices, validation_indices = next(
        splitter.split(features, labels, groups=groups)
    )
    if len(np.unique(labels[train_indices])) < 2:
        raise ValueError("Grouped training partition must contain both candidate labels")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        ),
    )
    started = time.perf_counter()
    model.fit(features[train_indices], labels[train_indices])

    validation_features = features[validation_indices]
    validation_labels = labels[validation_indices]
    probabilities = model.predict_proba(validation_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    validation_rows = candidates.iloc[validation_indices][
        ["trip_id", "observation_id", "label"]
    ].copy()
    validation_rows["probability"] = probabilities
    top_rows = validation_rows.loc[
        validation_rows.groupby(["trip_id", "observation_id"], sort=False)[
            "probability"
        ].idxmax()
    ]

    candidate_precision = float(
        precision_score(validation_labels, predictions, average="binary", zero_division=0)
    )
    candidate_recall = float(
        recall_score(validation_labels, predictions, average="binary", zero_division=0)
    )
    metrics = {
        "candidate_accuracy": float(accuracy_score(validation_labels, predictions)),
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "candidate_f1": float(
            f1_score(
                validation_labels,
                predictions,
                average="binary",
                zero_division=0,
            )
        ),
        "candidate_metric_average": "binary",
        "candidate_log_loss": float(log_loss(validation_labels, probabilities, labels=[0, 1])),
        "observation_top1_accuracy_with_candidates": float(top_rows["label"].mean()),
        "validation_candidate_rows": int(len(validation_indices)),
    }
    elapsed_seconds = time.perf_counter() - started

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        ),
    )
    final_model.fit(features, labels)
    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": final_model,
            "feature_columns": CANDIDATE_FEATURES,
            "candidate_generation": candidate_generation or {},
        },
        artifact_path,
    )

    return {
        "training_candidate_rows": int(len(candidates)),
        "training_trip_count": int(len(np.unique(groups))),
        "training_positive_candidates": int(labels.sum()),
        "training_negative_candidates": int(len(labels) - labels.sum()),
        "validation_trip_ids": sorted(np.unique(groups[validation_indices]).tolist()),
        "validation_metrics": metrics,
        "fit_time_seconds": elapsed_seconds,
        "model_path": str(artifact_path),
    }


def train_from_feature_directory(
    train_dir: str | Path,
    *,
    model_path: str | Path,
    summary_path: str | Path,
    top_k: int = 128,
    max_distance_m: float = 200.0,
    cell_size_m: float = 100.0,
    max_route_speed_mps: float = 80.0,
    start_tolerance_m: float = 200.0,
    end_tolerance_m: float = 600.0,
    advance_penalty_m: float = 5.0,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Generate labeled candidates from training trips and train the final model."""
    candidate_frames = []
    trip_summaries = []
    for trip_dir in iter_trip_directories(train_dir):
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
        if not candidates.empty:
            candidate_frames.append(candidates)
        trip_summaries.append(trip_summary)

    if not candidate_frames:
        raise ValueError(f"No candidate rows were generated from {train_dir}")

    training_candidates = pd.concat(candidate_frames, ignore_index=True)
    candidate_generation = {
        "top_k": top_k,
        "max_distance_m": max_distance_m,
        "cell_size_m": cell_size_m,
        "max_route_speed_mps": max_route_speed_mps,
        "route_start_tolerance_m": start_tolerance_m,
        "route_end_tolerance_m": end_tolerance_m,
        "route_alignment_advance_penalty_m": advance_penalty_m,
    }
    summary = train_logistic_regression(
        training_candidates,
        model_path=model_path,
        validation_size=validation_size,
        random_seed=random_seed,
        candidate_generation=candidate_generation,
    )
    validation_trip_ids = set(summary["validation_trip_ids"])
    validation_summaries = [
        item for item in trip_summaries if item["trip_id"] in validation_trip_ids
    ]
    validation_observations = sum(item["observations"] for item in validation_summaries)
    validation_hits = sum(
        item["candidate_hit_observations"] for item in validation_summaries
    )
    summary["validation_metrics"]["candidate_recall_pct"] = (
        100.0 * validation_hits / validation_observations
        if validation_observations
        else 0.0
    )
    summary["candidate_generation"] = {
        **candidate_generation,
        "trip_summaries": trip_summaries,
    }

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
