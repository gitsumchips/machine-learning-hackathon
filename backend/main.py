"""
PathSnap: FastAPI Backend for GPS Map Matching & Route Reconstruction.
Runs on Laptop 1.
Hosts the trained model, candidate generator, HMM/Viterbi decoder,
and server-side evaluation pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath("."))

from src.data_processing import load_trip
from src.hmm_decoder import decode_viterbi_route, compute_route_metrics, FEATURE_COLUMNS
from src.candidate_training import SegmentGridIndex
from src.utils.geo import (
    local_cartesian_projection,
    bearing_difference,
    haversine_distance_vectorized,
)

# Configure logging for live terminal audit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s"
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Distributed Map-Matching & Route Reconstruction API",
    description="Backend service hosting trained classifier and HMM/Viterbi sequential decoder for PES Hackathon evaluation.",
    version="2.0.0"
)

# Enable CORS for multi-laptop deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "Student folder"
if not DATA_DIR.exists():
    DATA_DIR = Path("data/Student folder")

MODEL = None

@app.on_event("startup")
def load_artifacts():
    global MODEL
    model_path = REPO_ROOT / "models" / "pathsnap_hist_gradient_boosting_tuned.joblib"
    if not model_path.exists():
        model_path = Path("models/pathsnap_hist_gradient_boosting_tuned.joblib")
    if not model_path.exists():
        model_path = REPO_ROOT / "PathSnap_GitHub_Artifacts" / "artifacts" / "hist_gradient_boosting" / "pathsnap_hist_gradient_boosting.joblib"
    logger.info(f"Loading trained classifier from {model_path}...")
    MODEL = joblib.load(model_path)
    logger.info("PathSnap model loaded successfully into memory. Ready for inference requests.")

@app.get("/")
def root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "Online",
        "model": "Tuned HistGradientBoosting + HMM/Viterbi Decoder",
        "device": "Laptop 1 (Model and API Server)"
    }

@app.get("/health")
@app.get("/health/")
@app.get("/api/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/trips")
@app.get("/trips/")
@app.get("/api/trips")
@app.get("/api/trips/")
def list_evaluation_trips():
    """List available evaluation trips."""
    test_dir = DATA_DIR / "student_test"
    if not test_dir.exists():
        logger.error(f"Student test directory not found at {test_dir.resolve()}")
        raise HTTPException(status_code=404, detail=f"Student test directory not found at {test_dir}")
    trips = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    logger.info(f"Listed {len(trips)} evaluation trips.")
    return {"trips": trips, "count": len(trips)}

def generate_candidates_for_inference(raw_trip) -> pd.DataFrame:
    """Generate candidate feature rows without accessing ground truth labels."""
    track_df = raw_trip.track
    nodes_df = raw_trip.nodes
    arcs_df = raw_trip.arcs

    ref_lon = float(track_df["longitude"].mean())
    ref_lat = float(track_df["latitude"].mean())

    track_x, track_y, _, _ = local_cartesian_projection(
        track_df["longitude"].to_numpy(), track_df["latitude"].to_numpy(),
        ref_lat=ref_lat, ref_lon=ref_lon
    )

    nodes_coords = nodes_df[["longitude", "latitude"]].to_numpy()
    nodes_x, nodes_y, _, _ = local_cartesian_projection(
        nodes_coords[:, 0], nodes_coords[:, 1],
        ref_lat=ref_lat, ref_lon=ref_lon
    )

    from_ids = arcs_df["from_node_id"].to_numpy(dtype=np.int64)
    to_ids = arcs_df["to_node_id"].to_numpy(dtype=np.int64)

    start_x = nodes_x[from_ids]
    start_y = nodes_y[from_ids]
    end_x = nodes_x[to_ids]
    end_y = nodes_y[to_ids]

    dx = end_x - start_x
    dy = end_y - start_y
    lengths = np.hypot(dx, dy)
    arc_bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    # Kinematics
    times = track_df["timestamp_s"].to_numpy()
    delta_time = np.r_[0.0, np.diff(times)]
    step_distance = np.zeros(len(track_df), dtype=float)
    if len(track_df) > 1:
        step_distance[1:] = haversine_distance_vectorized(
            track_df["longitude"].to_numpy()[:-1], track_df["latitude"].to_numpy()[:-1],
            track_df["longitude"].to_numpy()[1:], track_df["latitude"].to_numpy()[1:]
        )
    speed = np.divide(step_distance, delta_time, out=np.zeros_like(step_distance), where=delta_time > 0)
    accel = np.zeros(len(track_df), dtype=float)
    if len(track_df) > 1:
        accel[1:] = np.diff(speed) / np.maximum(delta_time[1:], 1e-3)

    gps_bearing = np.zeros(len(track_df), dtype=float)
    if len(track_df) > 1:
        gps_dx = track_x[1:] - track_x[:-1]
        gps_dy = track_y[1:] - track_y[:-1]
        gps_bearing[1:] = (np.degrees(np.arctan2(gps_dx, gps_dy)) + 360.0) % 360.0
        gps_bearing[0] = gps_bearing[1]
    heading_valid = (step_distance > 1e-4).astype(np.int8)

    out_degree = np.bincount(from_ids, minlength=len(nodes_df))
    in_degree = np.bincount(to_ids, minlength=len(nodes_df))

    arc_ids = np.arange(len(arcs_df), dtype=np.int64)
    index = SegmentGridIndex(arc_ids, start_x, start_y, end_x, end_y, cell_size_m=100.0)

    rows = []
    for obs_idx in range(len(track_df)):
        c_arcs, c_indices, dists, projs = index.query(
            track_x[obs_idx], track_y[obs_idx], top_k=8, max_distance_m=100.0
        )
        h_val = heading_valid[obs_idx]
        g_head = gps_bearing[obs_idx]

        for rank in range(len(c_arcs)):
            arc_id = c_arcs[rank]
            d_m = dists[rank]
            p_frac = projs[rank]
            a_head = arc_bearing[arc_id]
            s_len = lengths[arc_id]

            h_diff = 0.0
            if h_val:
                h_diff = bearing_difference(g_head, a_head)
            h_align = np.cos(np.radians(h_diff))

            rows.append({
                "observation_id": obs_idx,
                "arc_id": arc_id,
                "perpendicular_distance_m": d_m,
                "along_track_distance_m": p_frac * s_len,
                "segment_length_m": s_len,
                "gps_step_distance_m": step_distance[obs_idx],
                "gps_speed_mps": speed[obs_idx],
                "gps_acceleration_mps2": accel[obs_idx],
                "gps_delta_time_s": delta_time[obs_idx],
                "heading_alignment": h_align,
                "heading_valid": h_val,
                "arc_bearing_sin": np.sin(np.radians(a_head)),
                "arc_bearing_cos": np.cos(np.radians(a_head)),
                "from_out_degree": float(out_degree[from_ids[arc_id]]),
                "to_in_degree": float(in_degree[to_ids[arc_id]])
            })

    return pd.DataFrame(rows)

@app.post("/match-trip/{trip_id}")
@app.post("/match-trip/{trip_id}/")
@app.get("/match-trip/{trip_id}")
@app.post("/api/match-trip/{trip_id}")
@app.post("/api/match-trip/{trip_id}/")
@app.post("/match/{trip_id}")
@app.post("/match/{trip_id}/")
@app.post("/api/match/{trip_id}")
@app.post("/api/match/{trip_id}/")
def match_trip(trip_id: str):
    """
    Process evaluation trip through the complete prediction pipeline:
    Candidate Generation -> Model Scoring -> HMM/Viterbi Route Reconstruction -> Server-Side Metrics.
    """
    t_start = time.time()
    logger.info(f">>> INFERENCE REQUEST RECEIVED for Trip ID: {trip_id}")

    global MODEL
    if MODEL is None:
        load_artifacts()

    try:
        trip_dir = DATA_DIR / "student_test" / trip_id
        if not trip_dir.exists():
            logger.error(f"Trip {trip_id} not found in {trip_dir}")
            raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

        # 1. Load trip data (with server-side route labels strictly for evaluation)
        raw_trip = load_trip(trip_dir, include_route=True)
        ref_route = raw_trip.route.tolist() if raw_trip.route is not None else []
        logger.info(f"Loaded Trip {trip_id}: {len(raw_trip.track)} GPS points, {len(raw_trip.arcs)} arcs, {len(ref_route)} reference arcs.")

        # 2. Candidate generation (blind to reference route labels)
        t_cands = time.time()
        cands_df = generate_candidates_for_inference(raw_trip)
        logger.info(f"Generated {len(cands_df)} candidates in {time.time()-t_cands:.3f}s.")

        # 3. Model Scoring & HMM / Viterbi Route Reconstruction
        t_viterbi = time.time()
        pointwise_selected, connected_route = decode_viterbi_route(
            candidate_df=cands_df,
            nodes_df=raw_trip.nodes,
            arcs_df=raw_trip.arcs,
            model=MODEL,
            switch_penalty=40.0,
            gap_penalty=80.0,
            gap_scale_m=50.0
        )
        logger.info(f"HMM Viterbi decoding finished in {time.time()-t_viterbi:.3f}s. Reconstructed route length: {len(connected_route)} arcs.")

        # 4. Server-Side Accuracy Evaluation against internal reference route
        eval_metrics = compute_route_metrics(connected_route, ref_route, raw_trip.arcs)
        
        # Pointwise Top-1 accuracy against ground truth alignment
        from src.candidate_training import align_route_to_track
        track_coords = raw_trip.track[["longitude", "latitude"]].to_numpy()
        ref_lon, ref_lat = float(track_coords[:, 0].mean()), float(track_coords[:, 1].mean())
        tx, ty, _, _ = local_cartesian_projection(track_coords[:, 0], track_coords[:, 1], ref_lat=ref_lat, ref_lon=ref_lon)
        nodes_c = raw_trip.nodes[["longitude", "latitude"]].to_numpy()
        nx, ny, _, _ = local_cartesian_projection(nodes_c[:, 0], nodes_c[:, 1], ref_lat=ref_lat, ref_lon=ref_lon)
        from_ids = raw_trip.arcs["from_node_id"].to_numpy()
        to_ids = raw_trip.arcs["to_node_id"].to_numpy()
        gt_aligned = align_route_to_track(tx, ty, np.array(ref_route), np.arange(len(raw_trip.arcs)), nx[from_ids], ny[from_ids], nx[to_ids], ny[to_ids])
        
        pointwise_acc = float(np.mean(np.array(pointwise_selected) == gt_aligned))

        elapsed_ms = (time.time() - t_start) * 1000.0
        logger.info(f"=== EVALUATION RESULTS FOR TRIP {trip_id} ===")
        logger.info(f"Observation Top-1 Accuracy: {pointwise_acc*100:.2f}%")
        logger.info(f"Route Continuity:          {eval_metrics['route_continuity']*100:.2f}%")
        logger.info(f"Route Arc F1:              {eval_metrics['arc_f1']:.4f}")
        logger.info(f"Normalized Edit Similarity:{eval_metrics['normalized_edit_similarity']:.4f}")
        logger.info(f"Total Latency:             {elapsed_ms:.1f}ms")

        # 5. Prepare coordinate geometries for UI mapping
        nodes_lat_lon = raw_trip.nodes[["latitude", "longitude"]].to_numpy()
        
        pred_coords = []
        for arc_id in connected_route:
            u, v = from_ids[arc_id], to_ids[arc_id]
            pred_coords.append([float(nodes_lat_lon[u, 0]), float(nodes_lat_lon[u, 1])])
            pred_coords.append([float(nodes_lat_lon[v, 0]), float(nodes_lat_lon[v, 1])])

        ref_coords = []
        for arc_id in ref_route:
            u, v = from_ids[arc_id], to_ids[arc_id]
            ref_coords.append([float(nodes_lat_lon[u, 0]), float(nodes_lat_lon[u, 1])])
            ref_coords.append([float(nodes_lat_lon[v, 0]), float(nodes_lat_lon[v, 1])])

        # Subsample raw GPS for responsive payload
        step = max(1, len(raw_trip.track) // 250)
        gps_sub = raw_trip.track.iloc[::step][["latitude", "longitude", "timestamp_s"]].to_dict(orient="records")

        return {
            "status": "success",
            "trip_id": trip_id,
            "metrics": {
                "top1_accuracy": round(pointwise_acc, 4),
                "top1_accuracy_pct": round(pointwise_acc * 100.0, 2),
                "route_continuity": round(eval_metrics["route_continuity"], 4),
                "route_continuity_pct": round(eval_metrics["route_continuity"] * 100.0, 2),
                "route_arc_f1": round(eval_metrics["arc_f1"], 4),
                "route_edit_distance": eval_metrics["edit_distance"],
                "normalized_edit_similarity": round(eval_metrics["normalized_edit_similarity"], 4),
                "latency_ms": round(elapsed_ms, 1)
            },
            "predicted_route": connected_route,
            "predicted_route_length": len(connected_route),
            "reference_route_length": len(ref_route),
            "coordinates": {
                "predicted_route": pred_coords,
                "reference_route": ref_coords,
                "raw_gps": gps_sub
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error on trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error on trip {trip_id}: {str(e)}")
