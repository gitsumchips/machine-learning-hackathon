# PathSnap Model Comparison - Consolidated Table (Deliverables Section 3.1)

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student-Test Macro F1 | Top-1 Accuracy (%) | Route Continuity (%) | Log Loss | Inference Latency (ms/obs) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic Classifier | `C=1.0, solver=lbfgs, class_weight=balanced, max_iter=1000` | 0.5487 | 0.5449 | 68.21% | 48.24% | 0.3241 | 0.04 ms |
| **K-Nearest Neighbours (KNN)** | Instance-Based Non-Parametric | `n_neighbors=3, weights=uniform, metric=euclidean` | 0.6548 | 0.5284 | 65.48% | 51.18% | 0.4215 | 1.85 ms |
| **Random Forest** | Ensemble (Bagging Trees) | `n_estimators=150, max_features=0.7, min_samples_leaf=1` | 0.6669 | 0.6120 | 66.69% | 61.50% | 0.0700 | 0.82 ms |
| **Decision Tree** | Tree-Based (CART) | `criterion=gini, max_depth=14, min_samples_split=20, min_samples_leaf=10` | 0.7534 | 0.7684 | 70.43% | 62.84% | 0.2511 | 0.08 ms |
| **HistGradientBoosting (Baseline)** | Gradient Boosting (GBDT) | `lr=0.10, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2=1.0` | 0.7487 | 0.7701 | 77.17% | 70.29% | 0.1221 | 0.15 ms |
| **HistGradientBoosting (Tuned + HMM) [WINNER]** | Bayesian Tuned GBDT + HMM / Viterbi | `Optuna TPE: lr=0.0517, max_iter=125, max_leaf=61, min_leaf=25, l2=1.65; beta=0.10` | 0.9215 | 0.7749 | 81.84% | 99.87% | 0.3389 | 0.44 ms |
