"""KNN candidate ranker for GPS-to-road map matching.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Support both ``python -m src.knn_model`` and direct execution from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import TripData
from src.utils.geo import (
    batch_point_to_segment_projection,
    bearing_difference,
    initial_bearing_vectorized,
    local_cartesian_projection,
)


FEATURE_COLUMNS = [
    "distance_m",
    "heading_difference_deg",
    "log_arc_length_m",
    "projection_fraction",
    "speed_mps",
    "step_distance_m",
    "heading_valid",
]


def make_candidate_examples(
    trip: TripData,
    *,
    training: bool,
    max_candidates: int = 20,
    max_negatives: int = 8,
    max_observations: int | None = 500,
    random_state: int = 7,
) -> pd.DataFrame:
    """Create candidate-arc feature rows for one trip.

    Candidate search is approximate (nearest arc midpoints), followed by exact
    point-to-segment distance calculation in a local metric projection. During
    training, the nearest reference-route arc is added even if midpoint search
    did not return it. For speed, observations are evenly subsampled by default.
    """
    if max_candidates < 1 or max_negatives < 1:
        raise ValueError("max_candidates and max_negatives must be positive")
    if training and (trip.route is None or trip.route.empty):
        raise ValueError(f"{trip.trip_id}: training requires a non-empty reference route")
    if trip.track.empty or trip.arcs.empty or trip.nodes.empty:
        return pd.DataFrame(columns=["trip_id", "observation_id", "arc_id", *FEATURE_COLUMNS, "label", "target_arc_id"])

    track = trip.track.reset_index(drop=True)
    if max_observations is not None and len(track) > max_observations:
        observation_ids = np.linspace(0, len(track) - 1, max_observations, dtype=int)
    else:
        observation_ids = np.arange(len(track))

    gps_lon = track.longitude.to_numpy(dtype=float)
    gps_lat = track.latitude.to_numpy(dtype=float)
    node_lon = trip.nodes.longitude.to_numpy(dtype=float)
    node_lat = trip.nodes.latitude.to_numpy(dtype=float)
    all_lon = np.concatenate((gps_lon, node_lon))
    all_lat = np.concatenate((gps_lat, node_lat))
    ref_lon = float(np.mean(all_lon))
    ref_lat = float(np.mean(all_lat))
    gps_x, gps_y, kx, ky = local_cartesian_projection(gps_lon, gps_lat, ref_lat, ref_lon)
    node_x, node_y, _, _ = local_cartesian_projection(node_lon, node_lat, ref_lat, ref_lon)

    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    from_ids, to_ids = endpoints.T
    ax, ay = node_x[from_ids], node_y[from_ids]
    bx, by = node_x[to_ids], node_y[to_ids]
    midpoints = np.column_stack(((ax + bx) / 2.0, (ay + by) / 2.0))
    midpoint_tree = cKDTree(midpoints)
    candidate_count = min(max_candidates, len(trip.arcs))

    arc_bearing = initial_bearing_vectorized(
        trip.nodes.longitude.to_numpy()[from_ids],
        trip.nodes.latitude.to_numpy()[from_ids],
        trip.nodes.longitude.to_numpy()[to_ids],
        trip.nodes.latitude.to_numpy()[to_ids],
    )
    arc_length = np.hypot(bx - ax, by - ay)

    if training:
        route_ids = np.unique(trip.route.to_numpy(dtype=np.int64))
        route_ax, route_ay = ax[route_ids], ay[route_ids]
        route_bx, route_by = bx[route_ids], by[route_ids]
    else:
        route_ids = np.empty(0, dtype=np.int64)

    # Compute GPS motion features for all observations, then select the sampled rows.
    delta_time = np.r_[0.0, np.diff(track.timestamp_s.to_numpy(dtype=float))]
    step_distance = np.zeros(len(track), dtype=float)
    gps_heading = np.zeros(len(track), dtype=float)
    if len(track) > 1:
        from src.utils.geo import haversine_distance_vectorized

        step_distance[1:] = haversine_distance_vectorized(
            gps_lon[:-1], gps_lat[:-1], gps_lon[1:], gps_lat[1:]
        )
        gps_heading[1:] = initial_bearing_vectorized(
            gps_lon[:-1], gps_lat[:-1], gps_lon[1:], gps_lat[1:]
        )
    speed = np.divide(step_distance, delta_time, out=np.zeros_like(step_distance), where=delta_time > 0)
    heading_valid = (step_distance > 1e-6).astype(np.int8)

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for observation_id in observation_ids:
        point_x, point_y = float(gps_x[observation_id]), float(gps_y[observation_id])
        nearby = np.atleast_1d(midpoint_tree.query([point_x, point_y], k=candidate_count)[1]).astype(np.int64)

        target_arc_id = None
        if training:
            route_distances, _, _, _ = batch_point_to_segment_projection(
                point_x, point_y, route_ax, route_ay, route_bx, route_by
            )
            target_arc_id = int(route_ids[int(np.argmin(route_distances))])
            candidate_ids = np.unique(np.append(nearby, target_arc_id))
        else:
            candidate_ids = np.unique(nearby)

        distances, fractions, _, _ = batch_point_to_segment_projection(
            point_x, point_y, ax[candidate_ids], ay[candidate_ids], bx[candidate_ids], by[candidate_ids]
        )
        heading_diffs = bearing_difference(gps_heading[observation_id], arc_bearing[candidate_ids])
        positive_position = None
        if training:
            positive_position = int(np.flatnonzero(candidate_ids == target_arc_id)[0])
            negative_positions = np.flatnonzero(candidate_ids != target_arc_id)
            # Keep the closest alternatives; subsampling easy/distant negatives is unnecessary.
            order = np.argsort(distances[negative_positions])
            negative_positions = negative_positions[order[:max_negatives]]
            selected = np.concatenate(([positive_position], negative_positions))
        else:
            selected = np.arange(len(candidate_ids))

        for position in selected:
            arc_id = int(candidate_ids[position])
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "observation_id": int(observation_id),
                    "arc_id": arc_id,
                    "distance_m": float(distances[position]),
                    "heading_difference_deg": float(heading_diffs[position]),
                    "log_arc_length_m": float(np.log1p(arc_length[arc_id])),
                    "projection_fraction": float(fractions[position]),
                    "speed_mps": float(speed[observation_id]),
                    "step_distance_m": float(step_distance[observation_id]),
                    "heading_valid": int(heading_valid[observation_id]),
                    "label": int(training and arc_id == target_arc_id),
                    "target_arc_id": target_arc_id,
                }
            )

    return pd.DataFrame(rows)


def make_dataset(
    trips: Iterable[TripData],
    *,
    training: bool,
    max_observations: int | None = 500,
    max_candidates: int = 20,
    max_negatives: int = 8,
) -> pd.DataFrame:
    """Create candidate examples for a collection of trips."""
    frames = [
        make_candidate_examples(
            trip,
            training=training,
            max_observations=max_observations,
            max_candidates=max_candidates,
            max_negatives=max_negatives,
        )
        for trip in trips
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_knn_model(n_neighbors: int = 11, weights: str = "distance") -> object:
    """Create a scaled, distance-weighted KNN classifier."""
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm="kd_tree",
            leaf_size=40,
            n_jobs=-1,
        ),
    )


def rank_candidates(model: object, candidates: pd.DataFrame) -> pd.DataFrame:
    """Return one highest-scoring candidate arc per observation."""
    if candidates.empty:
        return pd.DataFrame(columns=["trip_id", "observation_id", "arc_id", "score"])
    probabilities = model.predict_proba(candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    classifier = model.steps[-1][1]
    positive_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
    ranked = candidates[["trip_id", "observation_id", "arc_id"]].copy()
    ranked["score"] = probabilities[:, positive_column]
    # Break equal KNN scores with geometric distance as a deterministic fallback.
    ranked["distance_m"] = candidates["distance_m"].to_numpy()
    ranked = ranked.sort_values(
        ["trip_id", "observation_id", "score", "distance_m"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return ranked.drop_duplicates(["trip_id", "observation_id"]).drop(columns="distance_m").reset_index(drop=True)


def decode_connected_route(
    model: object,
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    beam_width: int = 8,
    gap_scale_m: float = 25.0,
    gap_penalty: float = 2.0,
    switch_penalty: float = 80.0,
) -> list[int]:
    """Decode candidate scores into a graph-connected sequence of arc IDs.

    Viterbi chooses candidate arcs while preferring repeated or endpoint-
    connected arcs. Non-adjacent selected arcs are joined by a shortest path
    through the directed road graph; if no connector exists, the disconnected
    transition is kept and reported by route-level evaluation.
    """
    if candidates.empty:
        return []
    if beam_width < 1 or gap_scale_m <= 0:
        raise ValueError("beam_width and gap_scale_m must be positive")

    classifier = model.steps[-1][1]
    probabilities = model.predict_proba(candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    positive_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
    scored = candidates[["observation_id", "arc_id"]].copy()
    scored["score"] = probabilities[:, positive_column]
    scored = scored.sort_values(
        ["observation_id", "score"], ascending=[True, False], kind="stable"
    )
    groups = []
    for _, group in scored.groupby("observation_id", sort=True):
        group = group.head(beam_width)
        arc_ids = group.arc_id.to_numpy(dtype=np.int64)
        scores = group.score.to_numpy(dtype=float)
        # Normalize within each observation to make emission scales comparable.
        emissions = np.log(np.maximum(scores, 1e-6) / max(float(scores.sum()), 1e-6))
        groups.append((arc_ids, emissions))
    if not groups:
        return []

    edge_nodes = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    coordinates = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    from_ids, to_ids = edge_nodes.T
    lengths = np.asarray(
        [
            np.hypot(
                (coordinates[v, 0] - coordinates[u, 0]) * 111320.0 * np.cos(np.radians((coordinates[v, 1] + coordinates[u, 1]) / 2.0)),
                (coordinates[v, 1] - coordinates[u, 1]) * 110540.0,
            )
            for u, v in edge_nodes
        ],
        dtype=float,
    )
    forward: list[list[tuple[int, int, float]]] = [[] for _ in range(len(nodes))]
    reverse: list[list[tuple[int, int, float]]] = [[] for _ in range(len(nodes))]
    for arc_id, (u, v, length) in enumerate(zip(from_ids, to_ids, lengths)):
        forward[int(u)].append((int(v), arc_id, float(length)))
        reverse[int(v)].append((int(u), arc_id, float(length)))

    # Viterbi over per-observation candidate arcs.
    scores = groups[0][1].copy()
    backpointers = []
    for time_index in range(1, len(groups)):
        previous_ids, _ = groups[time_index - 1]
        current_ids, current_emissions = groups[time_index]
        previous_arcs = previous_ids[:, None]
        current_arcs = current_ids[None, :]
        previous_end = to_ids[previous_ids][:, None]
        current_start = from_ids[current_ids][None, :]
        same_arc = previous_arcs == current_arcs
        delta_lon = coordinates[current_start, 0] - coordinates[previous_end, 0]
        delta_lat = coordinates[current_start, 1] - coordinates[previous_end, 1]
        mean_lat = np.radians(
            (coordinates[current_start, 1] + coordinates[previous_end, 1]) / 2.0
        )
        gap = np.hypot(delta_lon * 111320.0 * np.cos(mean_lat), delta_lat * 110540.0)
        transition = -switch_penalty - gap / gap_scale_m
        transition -= (previous_end != current_start) * gap_penalty
        transition = np.where(same_arc, 0.0, transition)
        candidate_scores = scores[:, None] + transition
        back = np.argmax(candidate_scores, axis=0).astype(np.int64)
        next_scores = candidate_scores[back, np.arange(len(current_ids))]
        scores = next_scores + current_emissions
        backpointers.append(back)

    state = int(np.argmax(scores))
    selected = [int(groups[-1][0][state])]
    for time_index in range(len(groups) - 2, -1, -1):
        state = int(backpointers[time_index][state])
        selected.append(int(groups[time_index][0][state]))
    selected.reverse()

    # Collapse runs on the same road arc, then insert shortest directed
    # connectors where sampling skipped one or more road segments.
    compressed = [selected[0]]
    for arc_id in selected[1:]:
        if arc_id != compressed[-1]:
            compressed.append(arc_id)
    transitions = [
        (int(to_ids[left]), int(from_ids[right]))
        for left, right in zip(compressed[:-1], compressed[1:])
        if int(to_ids[left]) != int(from_ids[right])
    ]
    unique_sources = np.unique([source for source, _ in transitions]) if transitions else np.array([], dtype=np.int64)
    # Parallel directed arcs are represented by their shortest edge for graph
    # routing; keep its original arc row ID for reconstruction.
    shortest_edge_for_pair: dict[tuple[int, int], tuple[float, int]] = {}
    for arc_id, (source, target, length) in enumerate(zip(from_ids, to_ids, lengths)):
        key = (int(source), int(target))
        if key not in shortest_edge_for_pair or length < shortest_edge_for_pair[key][0]:
            shortest_edge_for_pair[key] = (float(length), arc_id)
    if shortest_edge_for_pair:
        rows, cols, data = zip(
            *((source, target, value[0]) for (source, target), value in shortest_edge_for_pair.items())
        )
        graph = coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    else:
        graph = coo_matrix((len(nodes), len(nodes))).tocsr()

    paths_by_pair: dict[tuple[int, int], list[int] | None] = {}
    targets_by_source: dict[int, list[int]] = {}
    for source, target in transitions:
        targets_by_source.setdefault(source, []).append(target)
    # Run C-optimized sparse Dijkstra in bounded source batches. This avoids
    # repeatedly traversing a large road graph in Python for each transition.
    source_batch_size = 32
    for batch_start in range(0, len(unique_sources), source_batch_size):
        source_batch = unique_sources[batch_start : batch_start + source_batch_size]
        _, predecessors = dijkstra(
            graph,
            directed=True,
            indices=source_batch,
            return_predecessors=True,
        )
        if predecessors.ndim == 1:
            predecessors = predecessors[np.newaxis, :]
        for row_index, source_value in enumerate(source_batch):
            source = int(source_value)
            predecessor = predecessors[row_index]
            for target in targets_by_source[source]:
                node = target
                reverse_arcs = []
                visited = set()
                while node != source:
                    if node in visited:
                        reverse_arcs = []
                        break
                    visited.add(node)
                    previous = int(predecessor[node])
                    if previous < 0:
                        reverse_arcs = []
                        break
                    edge = shortest_edge_for_pair.get((previous, node))
                    if edge is None:
                        reverse_arcs = []
                        break
                    reverse_arcs.append(edge[1])
                    node = previous
                paths_by_pair[(source, target)] = list(reversed(reverse_arcs)) if node == source else None

    connected = [compressed[0]]
    for next_arc in compressed[1:]:
        previous_end = int(to_ids[connected[-1]])
        next_start = int(from_ids[next_arc])
        bridge = paths_by_pair.get((previous_end, next_start), [])
        if bridge is not None:
            connected.extend(bridge)
        connected.append(next_arc)
    return connected


def route_sequence_metrics(predicted: Iterable[int], reference: Iterable[int]) -> dict[str, float]:
    """Compare a reconstructed arc sequence with the reference .route sequence."""
    pred = [int(value) for value in predicted]
    ref = [int(value) for value in reference]
    if not pred or not ref:
        raise ValueError("Predicted and reference routes must both be non-empty")

    # Levenshtein distance preserves ordering and accounts for skipped/repeated arcs.
    previous = list(range(len(ref) + 1))
    for i, pred_arc in enumerate(pred, start=1):
        current = [i]
        for j, ref_arc in enumerate(ref, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (pred_arc != ref_arc),
                )
            )
        previous = current
    edit_distance = previous[-1]

    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "route_edit_distance": int(edit_distance),
        "route_normalized_edit_similarity": float(1.0 - edit_distance / max(len(pred), len(ref))),
        "route_arc_precision": float(precision),
        "route_arc_recall": float(recall),
        "route_arc_f1": float(f1),
        "predicted_route_length": len(pred),
        "reference_route_length": len(ref),
    }


def evaluate_route_sequences(
    model: object,
    trips: Iterable[TripData],
    *,
    max_candidates: int = 20,
    beam_width: int = 8,
    switch_penalty: float = 80.0,
    max_observations: int | None = 500,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate connected reconstructed routes directly against each .route."""
    per_trip = []
    for trip in trips:
        if trip.route is None:
            raise ValueError(f"{trip.trip_id}: route labels are required for sequence evaluation")
        candidates = make_candidate_examples(
            trip,
            training=False,
            max_candidates=max_candidates,
            max_observations=max_observations,
        )
        predicted = decode_connected_route(
            model,
            candidates,
            trip.nodes,
            trip.arcs,
            beam_width=beam_width,
            switch_penalty=switch_penalty,
        )
        metrics = route_sequence_metrics(predicted, trip.route.to_numpy())
        if len(predicted) > 1:
            endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
            connected_transitions = sum(
                endpoints[left, 1] == endpoints[right, 0]
                for left, right in zip(predicted[:-1], predicted[1:])
            )
            metrics["connected_transition_fraction"] = float(
                connected_transitions / (len(predicted) - 1)
            )
        else:
            metrics["connected_transition_fraction"] = 1.0
        metrics["trip_id"] = trip.trip_id
        per_trip.append(metrics)
    frame = pd.DataFrame(per_trip)
    metric_columns = [
        "route_normalized_edit_similarity",
        "route_arc_precision",
        "route_arc_recall",
        "route_arc_f1",
        "connected_transition_fraction",
    ]
    aggregate = {f"mean_{column}": float(frame[column].mean()) for column in metric_columns}
    aggregate["trips_evaluated"] = int(len(frame))
    return aggregate, frame


def tune_route_decoder(
    model: object,
    validation_trips: Iterable[TripData],
    *,
    max_candidates: int = 20,
    max_observations: int | None = 500,
    switch_penalties: Iterable[float] = (20.0, 40.0, 60.0, 80.0, 120.0),
) -> pd.DataFrame:
    """Select temporal smoothing by reference-route sequence similarity.

    Candidate generation runs without route labels. Reference .route sequences
    are used only to score the decoded validation route.
    """
    trips = list(validation_trips)
    if not trips:
        raise ValueError("At least one validation trip is required")
    candidate_tables = {
        trip.trip_id: make_candidate_examples(
            trip,
            training=False,
            max_candidates=max_candidates,
            max_observations=max_observations,
        )
        for trip in trips
    }
    results = []
    for penalty in switch_penalties:
        if penalty < 0:
            raise ValueError("switch penalties must be non-negative")
        per_trip_scores = []
        for trip in trips:
            if trip.route is None:
                raise ValueError(f"{trip.trip_id}: route labels are required for tuning")
            predicted = decode_connected_route(
                model,
                candidate_tables[trip.trip_id],
                trip.nodes,
                trip.arcs,
                switch_penalty=float(penalty),
            )
            metrics = route_sequence_metrics(predicted, trip.route.to_numpy())
            per_trip_scores.append(metrics["route_normalized_edit_similarity"])
        results.append(
            {
                "switch_penalty": float(penalty),
                "mean_route_normalized_edit_similarity": float(np.mean(per_trip_scores)),
            }
        )
    return pd.DataFrame(results).sort_values(
        ["mean_route_normalized_edit_similarity", "switch_penalty"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def tune_knn_model(
    training_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    *,
    neighbor_values: Iterable[int] = (3, 5, 7, 11, 15, 21),
    weight_values: Iterable[str] = ("distance", "uniform"),
) -> tuple[object, pd.DataFrame]:
    """Select KNN neighborhood settings by proxy top-1 accuracy.

    The model is fitted once using the largest requested neighborhood size;
    validation scores are then compared across k and weighting options. The
    validation trips must be disjoint from the training trips.
    """
    neighbors = sorted(set(int(value) for value in neighbor_values))
    weights = list(dict.fromkeys(weight_values))
    if not neighbors or neighbors[0] < 1:
        raise ValueError("neighbor_values must contain positive integers")
    if not weights or any(value not in {"distance", "uniform"} for value in weights):
        raise ValueError("weight_values must contain 'distance' and/or 'uniform'")
    if training_examples.empty or validation_examples.empty:
        raise ValueError("Training and validation candidate tables must be non-empty")

    model = build_knn_model(n_neighbors=neighbors[-1])
    model.fit(
        training_examples[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        training_examples["label"].to_numpy(dtype=np.int8),
    )
    classifier = model.steps[-1][1]
    targets = validation_examples[
        ["trip_id", "observation_id", "target_arc_id"]
    ].drop_duplicates()
    results = []
    for weight in weights:
        for n_neighbors in neighbors:
            classifier.set_params(n_neighbors=n_neighbors, weights=weight)
            ranked = rank_candidates(model, validation_examples)
            checked = ranked.merge(
                targets,
                on=["trip_id", "observation_id"],
                validate="one_to_one",
            )
            results.append(
                {
                    "n_neighbors": n_neighbors,
                    "weights": weight,
                    "proxy_top1_accuracy": float(
                        np.mean(checked["arc_id"].to_numpy() == checked["target_arc_id"].to_numpy())
                    ),
                }
            )

    results_frame = pd.DataFrame(results).sort_values(
        ["proxy_top1_accuracy", "n_neighbors"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    best = results_frame.iloc[0]
    classifier.set_params(n_neighbors=int(best.n_neighbors), weights=str(best.weights))
    return model, results_frame


def score_proxy_labels(model: object, candidates: pd.DataFrame) -> dict[str, float]:
    """Evaluate candidate classification and top-1 accuracy against proxy labels."""
    if candidates.empty:
        raise ValueError("No candidate examples available for evaluation")
    prediction = model.predict(candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    predicted_route = rank_candidates(model, candidates)
    targets = candidates[["trip_id", "observation_id", "target_arc_id"]].drop_duplicates()
    checked = predicted_route.merge(targets, on=["trip_id", "observation_id"], validate="one_to_one")
    actual = candidates.label.to_numpy(dtype=np.int8)
    true_positive = int(((prediction == 1) & (actual == 1)).sum())
    predicted_positive = int((prediction == 1).sum())
    actual_positive = int((actual == 1).sum())
    return {
        "candidate_precision": float(true_positive / predicted_positive) if predicted_positive else 0.0,
        "candidate_recall": float(true_positive / actual_positive) if actual_positive else 0.0,
        "candidate_f1": float(2 * true_positive / (predicted_positive + actual_positive)) if predicted_positive + actual_positive else 0.0,
        "proxy_top1_accuracy": float(np.mean(checked.arc_id.to_numpy() == checked.target_arc_id.to_numpy())),
        "observations_evaluated": int(len(checked)),
    }


def main() -> None:
    """Tune, test, and save a KNN model from the local dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/Student folder")
    parser.add_argument("--output", default="results/knn/pathsnap_knn.joblib")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--n-neighbors",
        type=int,
        help="Optionally test only this k instead of searching the default k values.",
    )
    parser.add_argument("--max-observations", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    from src.data_processing import iter_trip_directories, load_trip

    trip_dirs = list(iter_trip_directories(Path(args.data_root) / "train"))
    trip_ids = [directory.name for directory in trip_dirs]
    if len(trip_ids) < 3:
        raise ValueError("At least three trips are required for fit/validation/test splits")
    if not 0 < args.validation_size < 1 or not 0 < args.test_size < 1:
        raise ValueError("--validation-size and --test-size must be between 0 and 1")
    if args.validation_size + args.test_size >= 1:
        raise ValueError("--validation-size plus --test-size must be less than 1")

    fit_validation_ids, test_ids = train_test_split(
        trip_ids, test_size=args.test_size, random_state=args.random_state
    )
    fit_ids, validation_ids = train_test_split(
        fit_validation_ids,
        test_size=args.validation_size,
        random_state=args.random_state + 1,
    )
    paths_by_id = {directory.name: directory for directory in trip_dirs}
    fit_trips = [load_trip(paths_by_id[trip_id], include_route=True) for trip_id in fit_ids]
    validation_trips = [
        load_trip(paths_by_id[trip_id], include_route=True) for trip_id in validation_ids
    ]
    test_trips = [load_trip(paths_by_id[trip_id], include_route=True) for trip_id in test_ids]
    dataset_options = {
        "max_observations": args.max_observations,
        "max_candidates": args.max_candidates,
        "max_negatives": args.max_negatives,
    }
    fit_examples = make_dataset(fit_trips, training=True, **dataset_options)
    validation_examples = make_dataset(validation_trips, training=True, **dataset_options)

    neighbor_values = (args.n_neighbors,) if args.n_neighbors else (3, 5, 7, 11, 15, 21)
    tuned_model, tuning_results = tune_knn_model(
        fit_examples,
        validation_examples,
        neighbor_values=neighbor_values,
    )
    selected = tuning_results.iloc[0]
    route_tuning_results = tune_route_decoder(
        tuned_model,
        validation_trips,
        max_candidates=args.max_candidates,
    )
    selected_switch_penalty = float(route_tuning_results.iloc[0].switch_penalty)
    test_metrics, per_trip_test_metrics = evaluate_route_sequences(
        tuned_model,
        test_trips,
        max_candidates=args.max_candidates,
        switch_penalty=selected_switch_penalty,
    )

    all_training_trips = fit_trips + validation_trips + test_trips
    all_examples = make_dataset(all_training_trips, training=True, **dataset_options)
    final_model = build_knn_model(
        n_neighbors=int(selected.n_neighbors), weights=str(selected.weights)
    )
    final_model.fit(
        all_examples[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        all_examples["label"].to_numpy(dtype=np.int8),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "n_neighbors": int(selected.n_neighbors),
            "weights": str(selected.weights),
            "max_candidates": args.max_candidates,
            "switch_penalty": selected_switch_penalty,
            "training_label_definition": "nearest reference-route arc by point-to-segment distance",
            "heldout_test_route_metrics": test_metrics,
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    report = {
        "selected_hyperparameters": {
            "n_neighbors": int(selected.n_neighbors),
            "weights": str(selected.weights),
        },
        "validation_tuning_scores": tuning_results.to_dict(orient="records"),
        "route_decoder_tuning_scores": route_tuning_results.to_dict(orient="records"),
        "heldout_test_route_metrics": test_metrics,
        "heldout_test_per_trip": per_trip_test_metrics.to_dict(orient="records"),
        "training_label_note": "Training still uses per-observation proxy labels derived by nearest point-to-segment distance among reference-route arcs.",
        "evaluation_note": "Held-out routes are reconstructed without route labels in candidate generation, then compared with each trip's supplied .route sequence.",
        "trip_counts": {
            "fit": len(fit_trips),
            "validation": len(validation_trips),
            "heldout_test": len(test_trips),
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2))

    print(f"Training candidate rows: {len(fit_examples):,}")
    print(f"Validation candidate rows: {len(validation_examples):,}")
    print("Hyperparameter search (validation proxy top-1):")
    print(tuning_results.to_string(index=False))
    print("Route decoder search (validation reference-route similarity):")
    print(route_tuning_results.to_string(index=False))
    print("Held-out route-sequence metrics (against supplied .route labels):")
    for name, value in test_metrics.items():
        print(f"  {name}: {value}")
    print(
        f"Selected model: k={int(selected.n_neighbors)}, weights={selected.weights}, "
        f"switch_penalty={selected_switch_penalty:g}"
    )
    print(f"Saved model: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()