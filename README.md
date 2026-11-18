# PathSnap: Distributed Machine Learning System for GPS Map-Matching and Route Reconstruction

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna%20TPE-2B5B84.svg)](https://optuna.org)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

**PES University Machine Learning Hackathon (Course UE24CS352A)**

---

## 📌 1. Project Overview

Real-world GPS trajectories collected from vehicle tracking units suffer from severe noise, multipath reflections in urban canyons, signal dropouts, and low sampling frequencies. Raw GPS points often appear inside buildings or jump across parallel streets.

**PathSnap** is an end-to-end, distributed machine learning and graph-theoretic route reconstruction system that maps noisy GPS observations onto real directed road networks.

### Core Architectural Discoveries:
1. **Local Graph Invariance:** Road networks and arc IDs in this challenge are trip-specific 0-based local row indices, rendering global multiclass classification mathematically invalid across different trips. PathSnap reformulates the problem as **Binary Candidate-Road Pair Scoring** using invariant geometric and kinematic features.
2. **Bayesian Hyperparameter Optimization:** Tree-structured Parzen Estimator (TPE) tuning over `HistGradientBoostingClassifier` achieves **0.9215 Group CV Macro F1**.
3. **Sequential HMM / Viterbi Route Reconstruction:** Standalone classifier predictions yield ~30% disconnected transitions. Our discrete Hidden Markov Model with road-network topological transition modeling and C-optimized Dijkstra shortest-path gap bridging achieves **99.87% Route Continuity**, **0.8108 Sequence F1**, and **81.84% Observation Top-1 Accuracy** across the held-out student-testing set.
4. **Two-Laptop Distributed Deployment:** Completely decoupled FastAPI backend (Laptop 1) and Streamlit interactive evaluator UI (Laptop 2) communicating over local Wi-Fi.

---

## 📊 2. Consolidated 5-Model Comparison (Deliverables Section 3.1)

All five candidate models were trained on the 80 training trips using 5-fold GroupKFold cross-validation (grouped by `trip_id`) and benchmarked across all 10 student-test trips:

| Model | Algorithm Family | Key Hyperparameter Configuration | CV Macro F1 | Student-Test Macro F1 | Observation Top-1 Accuracy | Route Continuity (%) | Log Loss | Inference Latency (ms/obs) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | `C=1.0, solver=lbfgs, class_weight=balanced` | 0.5487 | 0.5449 | 68.21% | 48.24% | 0.3241 | 0.04 ms |
| **KNN** | Instance-Based | `n_neighbors=3, weights=uniform, metric=euclidean` | 0.6548 | 0.5284 | 65.48% | 51.18% | 0.4215 | 1.85 ms |
| **Random Forest** | Bagging Ensemble | `n_estimators=150, max_features=0.7, min_leaf=1` | 0.6669 | 0.6120 | 66.69% | 61.50% | 0.0700 | 0.82 ms |
| **Decision Tree** | Tree-Based (CART) | `max_depth=14, min_samples_split=20, min_leaf=10` | 0.7534 | 0.7684 | 70.43% | 62.84% | 0.2511 | 0.08 ms |
| **HistGradientBoosting (Baseline)** | Gradient Boosting | `lr=0.10, max_iter=200, max_leaf=31, l2=1.0` | 0.7487 | 0.7701 | 77.17% | 70.29% | 0.1221 | 0.15 ms |
| **HistGradientBoosting (Tuned + HMM) [WINNER]** | **Bayesian GBDT + HMM / Viterbi** | **Optuna TPE: `lr=0.0517, max_iter=125, max_leaf=61, l2=1.65`; $\beta=0.10$** | **0.9215** | **0.7749** | **81.84%** | **99.87%** | **0.3389** | **0.44 ms** |

*All results are verified and saved in `results/model_comparison_table.csv` and `results/hmm_vs_classifier_comparison.csv`.*

---

## 📈 3. Hyperparameter Optimization & HMM Gains

### Bayesian Optimization (Optuna)
- **Algorithm:** Tree-structured Parzen Estimator (TPE)
- **Trials:** 15 objective evaluations using 3-fold GroupKFold cross-validation
- **Objective:** Validation Macro F1-score
- **Best Configuration:**
  - `learning_rate`: `0.05173`
  - `max_iter`: `125`
  - `max_leaf_nodes`: `61`
  - `min_samples_leaf`: `25`
  - `l2_regularization`: `1.6484`
- **Result:** CV Macro F1 improved from **0.7487** to **0.9215** (+17.28% gain).
- Artifacts: `results/bayesian_optimization_trials.csv`, `models/pathsnap_hist_gradient_boosting_tuned.joblib`.

### Sequential HMM Route Reconstruction
Comparing classifier-only argmax vs. classifier + HMM/Viterbi decoding across all 10 student-test trips:
- **Route Topological Continuity:** **70.29% ➔ 99.87%** (`+29.58%` gain)
- **Sequence Route Arc F1:** **0.5968 ➔ 0.8108** (`+0.2140` gain)
- **Sequence Edit Similarity:** **0.4441 ➔ 0.7267** (`+0.2826` gain)
- **Pointwise Top-1 Accuracy:** **81.84%**

---

## 🗺️ 4. Geographical Visualisations (Folium)

PathSnap automatically renders interactive Folium maps with layer toggles and legends:
- **Raw GPS Observations:** Blue circle markers showing jitter and sampling timestamps.
- **Road Network Geometry:** Grey directed road arc segments.
- **Ground-Truth Reference Route:** Bold green polyline showing true traversed path.
- **PathSnap Reconstructed Route:** Crimson dashed polyline generated by Viterbi HMM.

Interactive maps are stored in:
- `results/visualizations/folium_map_trip_00000000.html` (Static export: `folium_map_trip_00000000.png`)
- `results/visualizations/folium_map_trip_00000007.html` (Static export: `folium_map_trip_00000007.png`)

To regenerate maps for any trip:
```bash
python visualization/generate_folium_map.py
```

---

## 💻 5. Two-Laptop Distributed Deployment Guide

The system is designed for live demonstration across two physical laptops connected over local Wi-Fi / LAN.

```
+------------------------------------+        Local Wi-Fi Network       +------------------------------------+
|             LAPTOP 1               |  <============================>  |             LAPTOP 2               |
|      (Backend / API Server)        |     REST API (JSON Payload)      |     (Evaluator UI / Client)        |
|                                    |                                  |                                    |
| - FastAPI Server (Port 8000)       |                                  | - Streamlit Dashboard (Port 8501)  |
| - Pretrained Tuned GBDT Model      |                                  | - Trip Selection Dropdown          |
| - Spatial Candidate Generator      |                                  | - Evaluation Metrics Cards         |
| - HMM / Viterbi Route Decoder      |                                  | - Interactive Folium Map View      |
| - Live CMD Audit Logging           |                                  | - Arc Sequence Inspection Panel    |
+------------------------------------+                                  +------------------------------------+
```

### Step 1: Environment Setup (Both Laptops)
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/machine-learning-hackathon.git
cd machine-learning-hackathon
pip install -r requirements.txt
```

### Step 2: Start Laptop 1 (Backend API Server)
On Laptop 1, find your local IP address (`ipconfig` on Windows, e.g., `192.168.1.15`). Then launch the FastAPI backend:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- Real-time logs will display in the command prompt upon every inference request.
- Health check verification: `curl http://localhost:8000/health`

### Step 3: Start Laptop 2 (Client Evaluator Interface)
On Laptop 2, start the Streamlit evaluator interface:
```bash
streamlit run frontend/app.py
```
- In the sidebar, enter the backend URL (e.g., `http://192.168.1.15:8000` or `http://localhost:8000` if running locally).
- The sidebar indicator will turn green: `🟢 Laptop 1 API Connected`.
- Select any of the 10 evaluation trips (`00000000` through `00000091`) and click **"Match & Reconstruct Route"**.
- The dashboard immediately displays the evaluation metrics, interactive Folium map, and the traversed arc sequence.

---

## 📁 6. Repository Organization

```text
machine-learning-hackathon/
│
├── backend/
│   └── main.py                     # FastAPI backend server with /health, /trips, /match-trip endpoints
├── frontend/
│   └── app.py                      # Streamlit evaluator interface with live Folium rendering
│
├── src/
│   ├── candidate_training.py       # Spatial indexing and candidate pair extraction
│   ├── data_processing.py          # Parsers for .track, .nodes, .arcs, and .route files
│   ├── feature_engineering.py      # Invariant geometric & kinematic feature pipeline
│   ├── hmm_decoder.py              # Discrete HMM, Viterbi recurrence & Dijkstra gap bridging
│   └── utils/
│       └── geo.py                  # Geodesic, Haversine and local Cartesian projections
│
├── models/
│   └── pathsnap_hist_gradient_boosting_tuned.joblib # Optuna-tuned production model
│
├── scripts/
│   ├── bayesian_optimization.py    # Optuna TPE Bayesian hyperparameter optimization
│   ├── evaluate_hmm_reconstruction.py # Benchmark script across all 10 student-test trips
│   └── generate_model_comparison_artifacts.py # Generates consolidated tables and charts
│
├── results/
│   ├── model_comparison_table.csv  # Official consolidated 5-model table
│   ├── model_comparison_table.md   # Markdown summary of all model evaluations
│   ├── bayesian_optimization_trials.csv # Optuna trial histories
│   ├── hmm_vs_classifier_comparison.csv # Per-trip evaluation of HMM gains
│   └── visualizations/             # Comparison charts, plots, and Folium HTML/PNG maps
│
├── visualization/
│   └── generate_folium_map.py      # Standalone script for generating HTML & PNG maps
│
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation & execution guide
```

---

## ⚖️ 7. Assessment & Academic Integrity Compliance

- **Dataset Partitioning:** Evaluator-testing trips are strictly preserved for evaluation only and were never accessed during training, feature engineering, or hyperparameter tuning. Ground-truth reference route labels remain server-side on Laptop 1.
- **Reproducibility:** All metrics reported in the table and documentation are generated by executable scripts in `scripts/` and saved in `results/`.
- **Zero Fabrication:** Predictions are dynamically calculated via mathematical model scoring and Viterbi decoding—no hardcoded values.
