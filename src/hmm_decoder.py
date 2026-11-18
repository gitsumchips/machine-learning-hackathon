"""
HMM and Viterbi Route Reconstruction Module for PathSnap.
Implements hidden Markov modeling, emission scoring from tuned classifiers,
topological transition modeling, Viterbi recurrence, and route continuity evaluation.
"""

import os
import sys
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

FEATURE_COLUMNS = [
    'perpendicular_distance_m', 'along_track_distance_m', 'segment_length_m',
    'gps_step_distance_m', 'gps_speed_mps', 'gps_acceleration_mps2',
    'gps_delta_time_s', 'heading_alignment', 'heading_valid',
    'arc_bearing_sin', 'arc_bearing_cos', 'from_out_degree', 'to_in_degree'
]

def build_road_graph(arcs: pd.DataFrame, nodes: pd.DataFrame) -> Tuple[Any, Dict[Tuple[int, int], Tuple[float, int]]]:
    """
    Construct directed road network adjacency matrix for Dijkstra routing.
    """
    from_ids = arcs['from_node_id'].to_numpy(dtype=np.int64)
    to_ids = arcs['to_node_id'].to_numpy(dtype=np.int64)

    # Compute lengths if not present
    if 'length_m' in arcs.columns:
        lengths = arcs['length_m'].to_numpy(dtype=float)
    else:
        coords = nodes[['longitude', 'latitude']].to_numpy(dtype=float)
        # Approximate length in meters
        mid_lat = np.radians(coords[:, 1].mean())
        kx = 111320.0 * np.cos(mid_lat)
        ky = 110540.0
        dx = (coords[to_ids, 0] - coords[from_ids, 0]) * kx
        dy = (coords[to_ids, 1] - coords[from_ids, 1]) * ky
        lengths = np.hypot(dx, dy)

    shortest_edge_for_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for arc_id, (source, target, length) in enumerate(zip(from_ids, to_ids, lengths)):
        key = (int(source), int(target))
        if key not in shortest_edge_for_pair or length < shortest_edge_for_pair[key][0]:
            shortest_edge_for_pair[key] = (float(length), int(arc_id))

    num_nodes = len(nodes)
    if shortest_edge_for_pair:
        rows, cols, data = zip(
            *((source, target, value[0]) for (source, target), value in shortest_edge_for_pair.items())
        )
        graph = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    else:
        graph = coo_matrix((num_nodes, num_nodes)).tocsr()

    return graph, shortest_edge_for_pair

def decode_viterbi_route(
    candidate_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    model: Any,
    switch_penalty: float = 40.0,
    gap_penalty: float = 80.0,
    gap_scale_m: float = 50.0
) -> Tuple[List[int], List[int]]:
    """
    Run HMM Viterbi decoding over candidate observations.
    Returns:
        (pointwise_selected_arcs, full_connected_reconstructed_route)
    """
    # 1. Compute emission probabilities from model
    X = candidate_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]
    candidate_df = candidate_df.copy()
    candidate_df['prob'] = probs

    # Group candidates by observation_id
    obs_groups = []
    for obs_id, group in candidate_df.groupby('observation_id', sort=True):
        c_arc_ids = group['arc_id'].to_numpy(dtype=np.int64)
        c_probs = group['prob'].to_numpy(dtype=float)
        # Normalize emissions to avoid underflow
        p_sum = np.sum(c_probs)
        if p_sum > 0:
            c_norm = c_probs / p_sum
        else:
            c_norm = np.ones_like(c_probs) / len(c_probs)
        log_emissions = np.log(np.maximum(c_norm, 1e-6))
        obs_groups.append((c_arc_ids, log_emissions))

    if not obs_groups:
        return [], []

    from_ids = arcs_df['from_node_id'].to_numpy(dtype=np.int64)
    to_ids = arcs_df['to_node_id'].to_numpy(dtype=np.int64)
    coords = nodes_df[['longitude', 'latitude']].to_numpy(dtype=float)
    mean_lat = np.radians(coords[:, 1].mean())
    kx = 111320.0 * np.cos(mean_lat)
    ky = 110540.0

    # 2. Viterbi Forward Recurrence
    scores = obs_groups[0][1].copy()
    backpointers = []

    for t in range(1, len(obs_groups)):
        prev_arcs, _ = obs_groups[t - 1]
        curr_arcs, curr_emissions = obs_groups[t]

        prev_ends = to_ids[prev_arcs][:, None]
        curr_starts = from_ids[curr_arcs][None, :]

        same_arc = (prev_arcs[:, None] == curr_arcs[None, :])

        dx = (coords[curr_starts, 0] - coords[prev_ends, 0]) * kx
        dy = (coords[curr_starts, 1] - coords[prev_ends, 1]) * ky
        gap = np.hypot(dx, dy)

        transition = -switch_penalty - gap / gap_scale_m
        transition -= (prev_ends != curr_starts) * gap_penalty
        transition = np.where(same_arc, 0.0, transition)

        candidate_matrix = scores[:, None] + transition
        best_prev = np.argmax(candidate_matrix, axis=0)
        best_scores = candidate_matrix[best_prev, np.arange(len(curr_arcs))]
        scores = best_scores + curr_emissions
        backpointers.append(best_prev)

    # 3. Backtracking
    state = int(np.argmax(scores))
    selected = [int(obs_groups[-1][0][state])]
    for t in range(len(obs_groups) - 2, -1, -1):
        state = int(backpointers[t][state])
        selected.append(int(obs_groups[t][0][state]))
    selected.reverse()

    # 4. Topological Gap Bridging via Dijkstra
    graph, shortest_edges = build_road_graph(arcs_df, nodes_df)

    # Collapse consecutive identical arcs
    compressed = [selected[0]]
    for arc_id in selected[1:]:
        if arc_id != compressed[-1]:
            compressed.append(arc_id)

    # Find transitions requiring topological bridging
    transitions = []
    for left, right in zip(compressed[:-1], compressed[1:]):
        u_end = int(to_ids[left])
        v_start = int(from_ids[right])
        if u_end != v_start:
            transitions.append((u_end, v_start))

    unique_sources = list(set(s for s, _ in transitions))
    paths_by_pair = {}
    if unique_sources:
        for s in unique_sources:
            _, preds = dijkstra(graph, directed=True, indices=[s], return_predecessors=True)
            preds = preds[0]
            targets = [t for src, t in transitions if src == s]
            for t in targets:
                node = t
                rev_path = []
                visited = set()
                while node != s and node >= 0 and node not in visited:
                    visited.add(node)
                    p = int(preds[node])
                    if p >= 0 and (p, node) in shortest_edges:
                        rev_path.append(shortest_edges[(p, node)][1])
                        node = p
                    else:
                        break
                if node == s:
                    paths_by_pair[(s, t)] = list(reversed(rev_path))
                else:
                    paths_by_pair[(s, t)] = []

    connected_route = [compressed[0]]
    for next_arc in compressed[1:]:
        prev_end = int(to_ids[connected_route[-1]])
        next_start = int(from_ids[next_arc])
        if prev_end != next_start:
            bridge = paths_by_pair.get((prev_end, next_start), [])
            connected_route.extend(bridge)
        connected_route.append(next_arc)

    return selected, connected_route

def compute_route_metrics(predicted: List[int], reference: List[int], arcs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Levenshtein edit distance, normalized similarity, precision, recall, F1,
    and route continuity (% connected transitions).
    """
    pred = [int(x) for x in predicted]
    ref = [int(x) for x in reference]

    if not pred or not ref:
        return {"error": "Empty route"}

    # Levenshtein distance
    m, n = len(ref), len(pred)
    prev = list(range(m + 1))
    for i, p_arc in enumerate(pred, 1):
        curr = [i]
        for j, r_arc in enumerate(ref, 1):
            cost = 0 if p_arc == r_arc else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    edit_dist = prev[-1]
    edit_sim = max(0.0, 1.0 - (edit_dist / max(m, n)))

    # Arc-level overlap F1
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    prec = overlap / len(pred)
    rec = overlap / len(ref)
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Route Continuity: check consecutive transitions in arcs_df
    to_ids = arcs_df['to_node_id'].to_numpy(dtype=np.int64)
    from_ids = arcs_df['from_node_id'].to_numpy(dtype=np.int64)
    connected_steps = 0
    total_steps = max(len(pred) - 1, 1)
    for i in range(len(pred) - 1):
        a1, a2 = pred[i], pred[i + 1]
        if a1 == a2 or to_ids[a1] == from_ids[a2]:
            connected_steps += 1
    continuity = connected_steps / total_steps

    return {
        "edit_distance": int(edit_dist),
        "normalized_edit_similarity": float(edit_sim),
        "arc_precision": float(prec),
        "arc_recall": float(rec),
        "arc_f1": float(f1),
        "route_continuity": float(continuity),
        "predicted_route_length": len(pred),
        "reference_route_length": len(ref)
    }
