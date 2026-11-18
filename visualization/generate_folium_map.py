"""
Generate interactive Folium maps comparing original GPS observations,
road network geometry, reference route, and HMM predicted route.
"""

import os
import sys
import folium
from folium import plugins
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from src.data_processing import load_trip
from src.hmm_decoder import decode_viterbi_route, FEATURE_COLUMNS

def create_folium_map(trip_id: str = "00000000"):
    print(f"Generating Folium map for Trip {trip_id}...")
    student_test_dir = Path("data/Student folder/student_test")
    raw_trip = load_trip(student_test_dir / trip_id, include_route=True)

    # Load candidates & model
    model = joblib.load("models/pathsnap_hist_gradient_boosting_tuned.joblib")
    df_test = pd.read_parquet("results/student_test_candidates_generated.parquet")
    trip_cands = df_test[df_test["trip_id"] == trip_id].copy().reset_index(drop=True)

    # Decode route
    _, predicted_route = decode_viterbi_route(
        candidate_df=trip_cands,
        nodes_df=raw_trip.nodes,
        arcs_df=raw_trip.arcs,
        model=model,
        switch_penalty=40.0,
        gap_penalty=80.0,
        gap_scale_m=50.0
    )

    ref_route = raw_trip.route.tolist()
    track_df = raw_trip.track
    nodes_df = raw_trip.nodes
    arcs_df = raw_trip.arcs

    center_lat = float(track_df["latitude"].mean())
    center_lon = float(track_df["longitude"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # 1. Road Network Layer (subsampled bounding box)
    lat_min, lat_max = track_df["latitude"].min() - 0.005, track_df["latitude"].max() + 0.005
    lon_min, lon_max = track_df["longitude"].min() - 0.005, track_df["longitude"].max() + 0.005

    network_group = folium.FeatureGroup(name="Road Network (Arcs)", show=True)
    coords = nodes_df[["latitude", "longitude"]].to_numpy()
    from_ids = arcs_df["from_node_id"].to_numpy()
    to_ids = arcs_df["to_node_id"].to_numpy()

    # Draw relevant arcs within bounding box
    for i in range(len(arcs_df)):
        u, v = from_ids[i], to_ids[i]
        u_lat, u_lon = coords[u]
        v_lat, v_lon = coords[v]
        if (lat_min <= u_lat <= lat_max and lon_min <= u_lon <= lon_max) or \
           (lat_min <= v_lat <= lat_max and lon_min <= v_lon <= lon_max):
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#888888",
                weight=1.5,
                opacity=0.4
            ).add_to(network_group)
    network_group.add_to(m)

    # 2. Reference Route (Ground Truth) - Bold Green
    ref_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
    ref_coords = []
    for arc_id in ref_route:
        u, v = from_ids[arc_id], to_ids[arc_id]
        ref_coords.append([coords[u][0], coords[u][1]])
        ref_coords.append([coords[v][0], coords[v][1]])
    folium.PolyLine(
        locations=ref_coords,
        color="#2E7D32",
        weight=5.0,
        opacity=0.8,
        tooltip="Ground-Truth Traversed Route"
    ).add_to(ref_group)
    ref_group.add_to(m)

    # 3. Predicted Reconstructed Route (HMM) - Bold Crimson
    pred_group = folium.FeatureGroup(name="Predicted Route (PathSnap HMM)", show=True)
    pred_coords = []
    for arc_id in predicted_route:
        u, v = from_ids[arc_id], to_ids[arc_id]
        pred_coords.append([coords[u][0], coords[u][1]])
        pred_coords.append([coords[v][0], coords[v][1]])
    folium.PolyLine(
        locations=pred_coords,
        color="#D32F2F",
        weight=3.5,
        opacity=0.9,
        dash_array="6, 6",
        tooltip="Predicted Reconstructed Route"
    ).add_to(pred_group)
    pred_group.add_to(m)

    # 4. Raw GPS Observations - Blue Dots
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    # Subsample for smooth rendering
    step = max(1, len(track_df) // 300)
    for idx in range(0, len(track_df), step):
        row = track_df.iloc[idx]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3.5,
            color="#1976D2",
            fill=True,
            fill_color="#2196F3",
            fill_opacity=0.8,
            popup=f"Obs {idx} | t={row['timestamp_s']:.1f}s"
        ).add_to(gps_group)
    gps_group.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    # Floating Legend HTML
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 260px; height: 160px; 
                background-color: white; border:2px solid #555; z-index:9999; 
                font-size:13px; font-family:sans-serif; padding: 10px; border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <b>PathSnap Map Matching Layers</b><br>
    <i style="background:#2196F3; width:10px; height:10px; display:inline-block; border-radius:50%; margin-right:6px;"></i> Raw GPS Observations<br>
    <i style="background:#888888; width:16px; height:3px; display:inline-block; margin-right:6px;"></i> Road Network (Arcs)<br>
    <i style="background:#2E7D32; width:16px; height:4px; display:inline-block; margin-right:6px;"></i> Reference Route (Ground Truth)<br>
    <i style="background:#D32F2F; width:16px; height:4px; display:inline-block; margin-right:6px; border-top: 1px dashed white;"></i> Predicted Route (HMM Decoder)<br>
    <hr style="margin:5px 0;">
    <small>PES University ML Hackathon</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs("results/visualizations", exist_ok=True)
    html_out = f"results/visualizations/folium_map_trip_{trip_id}.html"
    m.save(html_out)
    print(f"Saved interactive Folium map to {html_out}")

    # Generate Matplotlib Static Screenshot / Plot as well for PDF report embedding
    plt.figure(figsize=(10, 8))
    # Subsample GPS
    gps_sub = track_df.iloc[::step]
    plt.scatter(gps_sub["longitude"], gps_sub["latitude"], color="#1976D2", s=15, alpha=0.7, label="Raw GPS Points (Noisy)", zorder=3)

    # Reference route
    ref_lats = [pt[0] for pt in ref_coords]
    ref_lons = [pt[1] for pt in ref_coords]
    plt.plot(ref_lons, ref_lats, color="#2E7D32", linewidth=3.5, alpha=0.85, label="Reference Route (Ground Truth)", zorder=4)

    # Predicted route
    pred_lats = [pt[0] for pt in pred_coords]
    pred_lons = [pt[1] for pt in pred_coords]
    plt.plot(pred_lons, pred_lats, color="#D32F2F", linewidth=2.0, linestyle="--", alpha=0.9, label="Predicted Route (PathSnap HMM)", zorder=5)

    plt.title(f"PathSnap Map-Matching & Route Reconstruction (Trip {trip_id})", fontsize=13, fontweight="bold")
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    png_out = f"results/visualizations/folium_map_trip_{trip_id}.png"
    plt.savefig(png_out, dpi=200)
    plt.close()
    print(f"Saved static visualization plot to {png_out}")

if __name__ == "__main__":
    create_folium_map("00000000")
    create_folium_map("00000007")
