"""
PathSnap: Streamlit Frontend Client for Evaluator Interface (Laptop 2).
Connects to Laptop 1 (FastAPI backend) over local Wi-Fi / LAN.
Visualizes map-matching results, metrics, and interactive Folium maps.
"""

import time
import requests
import streamlit as st
import folium
from folium import plugins
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap | GPS Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished hackathon presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-badge {
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/map-editing.png", width=64)
st.sidebar.title("PathSnap Client (Laptop 2)")
st.sidebar.markdown("**PES University ML Hackathon**")
st.sidebar.markdown("---")

backend_url = st.sidebar.text_input(
    "FastAPI Backend URL (Laptop 1):",
    value="http://localhost:8000",
    help="Enter the IP address of Laptop 1, e.g., http://192.168.1.15:8000"
).rstrip("/")

# Check Backend Status
backend_online = False
trips_list = ["00000000", "00000007", "00000009", "00000040", "00000052", "00000056", "00000065", "00000075", "00000083", "00000088"]

try:
    health_resp = requests.get(f"{backend_url}/health", timeout=2)
    if health_resp.status_code == 200:
        backend_online = True
        st.sidebar.success("🟢 Laptop 1 API Connected")
        
        # Fetch trips from backend
        trips_resp = requests.get(f"{backend_url}/trips", timeout=2)
        if trips_resp.status_code == 200:
            fetched_trips = trips_resp.json().get("trips", [])
            if fetched_trips:
                trips_list = fetched_trips
    else:
        st.sidebar.warning(f"🟡 API responded with code {health_resp.status_code}")
except Exception as e:
    st.sidebar.error("🔴 Backend Offline / Unreachable")
    st.sidebar.caption(f"Cannot reach `{backend_url}`. Start Laptop 1 with: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`")

st.sidebar.markdown("---")
st.sidebar.subheader("Select Evaluation Trip")
selected_trip = st.sidebar.selectbox("Trip ID:", trips_list, index=0)

run_button = st.sidebar.button("🚀 Match & Reconstruct Route", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- **Classifier:** HistGradientBoosting (Optuna-tuned)
- **Sequential Model:** Discrete HMM (Viterbi Decoding)
- **Topological Smoother:** Dijkstra Shortest-Path Bridging
- **Laptop 1:** Model Hosting & Graph API
- **Laptop 2:** Interactive Client & Evaluator Dashboard
""")

# Main Content
st.markdown('<div class="main-header">PathSnap GPS Map-Matching System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Distributed Machine Learning & Topological Route Reconstruction across Road Networks</div>', unsafe_allow_html=True)

if not run_button:
    # Landing / Overview page
    st.info("👈 Select an evaluation trip from the sidebar and click **'Match & Reconstruct Route'** to run the complete distributed inference pipeline.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Model Performance Summary (Student Test Set)")
        st.markdown("""
        - **Top-1 Observation Accuracy:** `81.84%`
        - **Route Topological Continuity:** `99.87%` *(+29.58% over classifier alone)*
        - **Sequence Route Arc F1:** `0.8108` *(+0.2140 over classifier alone)*
        - **Sequence Edit Similarity:** `0.7267` *(+0.2826 over classifier alone)*
        - **Inference Latency:** `< 0.5 ms` per GPS observation
        """)
    with col2:
        st.subheader("Key Algorithmic Innovations")
        st.markdown("""
        1. **Invariant Feature Geometry:** Resolves local arc IDs by projecting coordinates into invariant perpendicular distances, heading alignment cosines, and topological graph degrees.
        2. **Bayesian Hyperparameter Tuning:** Tree-structured Parzen Estimator (TPE) via Optuna achieved **0.9215 Group CV Macro F1**.
        3. **HMM Viterbi Decoder:** Combines calibrated classifier emission probabilities with road-network topological transition penalties.
        4. **Dijkstra Gap-Bridging:** Guarantees 99.87% connected vehicle routes across noisy urban canyons.
        """)

else:
    # Run prediction request
    progress_bar = st.progress(10, text="Sending inference request to Laptop 1...")
    t0 = time.time()
    
    try:
        req_url = f"{backend_url}/match-trip/{selected_trip}"
        response = requests.post(req_url, timeout=60)
        
        # Fallback to alternate route aliases if 404
        if response.status_code == 404:
            for alt_endpoint in [f"{backend_url}/api/match-trip/{selected_trip}", f"{backend_url}/match/{selected_trip}"]:
                alt_resp = requests.post(alt_endpoint, timeout=60)
                if alt_resp.status_code == 200:
                    response = alt_resp
                    req_url = alt_endpoint
                    break

        progress_bar.progress(60, text="Decoding HMM Viterbi sequence...")
        
        if response.status_code != 200:
            progress_bar.empty()
            st.error(f"Inference failed with status {response.status_code} at `{req_url}`: {response.text}")
            if response.status_code == 404:
                st.warning("⚠️ **Backend 404 Not Found Error:** Laptop 1 is running an older version of `backend/main.py` that does not have the `/match-trip` route registered. Please restart Uvicorn on Laptop 1 with: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`")
        else:
            progress_bar.progress(90, text="Rendering interactive map layers...")
            data = response.json()
            metrics = data["metrics"]
            coords = data["coordinates"]
            pred_route = data["predicted_route"]
            progress_bar.progress(100, text="Complete!")
            progress_bar.empty()
            
            # 1. Performance Metric Cards
            st.subheader(f"Evaluation Metrics for Trip `{selected_trip}`")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Top-1 Accuracy</div>
                    <div class="metric-val">{metrics['top1_accuracy_pct']}%</div>
                    <span class="metric-badge">Observation-level</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #10B981;">
                    <div class="metric-title">Route Continuity</div>
                    <div class="metric-val">{metrics['route_continuity_pct']}%</div>
                    <span class="metric-badge" style="background:#D1FAE5; color:#065F46;">Connected Path</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #8B5CF6;">
                    <div class="metric-title">Sequence Arc F1</div>
                    <div class="metric-val">{metrics['route_arc_f1']:.4f}</div>
                    <span class="metric-badge" style="background:#EDE9FE; color:#5B21B6;">Traversal F1</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col4:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #F59E0B;">
                    <div class="metric-title">Edit Similarity</div>
                    <div class="metric-val">{metrics['normalized_edit_similarity']:.4f}</div>
                    <span class="metric-badge" style="background:#FEF3C7; color:#92400E;">Levenshtein Sim</span>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col5:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #EF4444;">
                    <div class="metric-title">Latency</div>
                    <div class="metric-val">{metrics['latency_ms']:.1f} ms</div>
                    <span class="metric-badge" style="background:#FEE2E2; color:#991B1B;">End-to-End</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2. Interactive Map Section
            st.subheader("Geographical Map Visualization")
            
            # Compute center from raw GPS
            raw_gps = coords.get("raw_gps", [])
            if raw_gps:
                center_lat = sum(p["latitude"] for p in raw_gps) / len(raw_gps)
                center_lon = sum(p["longitude"] for p in raw_gps) / len(raw_gps)
            else:
                center_lat, center_lon = 12.9716, 77.5946
                
            # Create Folium Map
            fmap = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles="OpenStreetMap"
            )
            
            # Layer 1: Raw GPS Points
            gps_layer = folium.FeatureGroup(name="Raw Noisy GPS Observations", show=True)
            for p in raw_gps:
                folium.CircleMarker(
                    location=[p["latitude"], p["longitude"]],
                    radius=3,
                    color="#1d4ed8",
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.7,
                    tooltip=f"GPS point @ t={p.get('timestamp_s', 0):.0f}s"
                ).add_to(gps_layer)
            gps_layer.add_to(fmap)
            
            # Layer 2: Ground Truth Reference Route
            ref_pts = coords.get("reference_route", [])
            if ref_pts:
                ref_layer = folium.FeatureGroup(name="Ground-Truth Reference Route (Green)", show=True)
                folium.PolyLine(
                    locations=ref_pts,
                    color="#16a34a",
                    weight=5.5,
                    opacity=0.85,
                    tooltip="Ground Truth Reference Route"
                ).add_to(ref_layer)
                ref_layer.add_to(fmap)
                
            # Layer 3: Reconstructed Predicted Route
            pred_pts = coords.get("predicted_route", [])
            if pred_pts:
                pred_layer = folium.FeatureGroup(name="Reconstructed Predicted Route (Crimson)", show=True)
                folium.PolyLine(
                    locations=pred_pts,
                    color="#dc2626",
                    weight=4.0,
                    opacity=0.9,
                    dash_array="6, 8",
                    tooltip="PathSnap Reconstructed Route (HMM Viterbi)"
                ).add_to(pred_layer)
                pred_layer.add_to(fmap)
                
            # Layer control & Legend
            folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
            
            legend_html = """
            <div style="position: fixed; 
                        bottom: 30px; left: 30px; width: 260px; height: 120px; 
                        background-color: white; border:2px solid #ccc; z-index:9999; 
                        font-size:12px; padding: 10px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
                <b>PathSnap Route Legend</b><br>
                <i style="background: #3b82f6; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Raw GPS Observations<br>
                <i style="background: #16a34a; width: 20px; height: 4px; display: inline-block; margin-right: 6px;"></i> Reference Route (Ground Truth)<br>
                <i style="border-top: 3px dashed #dc2626; width: 20px; display: inline-block; margin-right: 6px; vertical-align: middle;"></i> Predicted Route (Viterbi HMM)<br>
                <small style="color: gray;">Trip: <b>{}</b></small>
            </div>
            """.format(selected_trip)
            fmap.get_root().html.add_child(folium.Element(legend_html))
            
            # Embed Folium map in Streamlit
            map_html = fmap.get_root().render()
            components.html(map_html, height=560, scrolling=False)
            
            # 3. Reconstructed Sequence Inspection
            st.markdown("<br>", unsafe_allow_html=True)
            exp_col1, exp_col2 = st.columns([1, 1])
            with exp_col1:
                st.subheader(f"Reconstructed Arc Sequence ({len(pred_route)} Arcs)")
                st.write(f"Traversed Road Arc IDs (in chronological sequence):")
                st.code(str(pred_route[:60]) + ("..." if len(pred_route) > 60 else ""), language="python")
                
            with exp_col2:
                st.subheader("Route Traversal Diagnostics")
                st.write(f"- **Total Reconstructed Arcs:** `{data['predicted_route_length']}`")
                st.write(f"- **Reference Route Arcs:** `{data['reference_route_length']}`")
                st.write(f"- **Length Ratio:** `{data['predicted_route_length'] / max(1, data['reference_route_length']):.3f}`")
                st.write(f"- **Graph Edit Distance:** `{metrics['route_edit_distance']}`")
                
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to Laptop 1 backend at `{backend_url}`.")
        st.info("Check that Laptop 1 is running `uvicorn backend.main:app --host 0.0.0.0 --port 8000` and that both laptops are on the same Wi-Fi network.")
    except Exception as ex:
        st.error(f"Unexpected error: {str(ex)}")
