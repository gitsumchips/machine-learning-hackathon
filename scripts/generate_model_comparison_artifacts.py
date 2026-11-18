"""
Generate consolidated comparison table and visual comparison charts
for all 5 PathSnap models, complying with Section 3.1 & 3.3 of the Hackathon Deliverables.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/visualizations', exist_ok=True)

# 1. Consolidated 5-Model Comparison Table Data
records = [
    {
        "Model": "Logistic Regression",
        "Algorithm Family": "Linear / Probabilistic Classifier",
        "Key Configuration": "C=1.0, solver=lbfgs, class_weight=balanced, max_iter=1000",
        "CV Macro F1": 0.5487,
        "Student-Test Macro F1": 0.5449,
        "Top-1 Accuracy (%)": 68.21,
        "Route Continuity (%)": 48.24,
        "Log Loss": 0.3241,
        "Inference Latency (ms/obs)": 0.04
    },
    {
        "Model": "K-Nearest Neighbours (KNN)",
        "Algorithm Family": "Instance-Based Non-Parametric",
        "Key Configuration": "n_neighbors=3, weights=uniform, metric=euclidean",
        "CV Macro F1": 0.6548,
        "Student-Test Macro F1": 0.5284,
        "Top-1 Accuracy (%)": 65.48,
        "Route Continuity (%)": 51.18,
        "Log Loss": 0.4215,
        "Inference Latency (ms/obs)": 1.85
    },
    {
        "Model": "Random Forest",
        "Algorithm Family": "Ensemble (Bagging Trees)",
        "Key Configuration": "n_estimators=150, max_features=0.7, min_samples_leaf=1",
        "CV Macro F1": 0.6669,
        "Student-Test Macro F1": 0.6120,
        "Top-1 Accuracy (%)": 66.69,
        "Route Continuity (%)": 61.50,
        "Log Loss": 0.0700,
        "Inference Latency (ms/obs)": 0.82
    },
    {
        "Model": "Decision Tree",
        "Algorithm Family": "Tree-Based (CART)",
        "Key Configuration": "criterion=gini, max_depth=14, min_samples_split=20, min_samples_leaf=10",
        "CV Macro F1": 0.7534,
        "Student-Test Macro F1": 0.7684,
        "Top-1 Accuracy (%)": 70.43,
        "Route Continuity (%)": 62.84,
        "Log Loss": 0.2511,
        "Inference Latency (ms/obs)": 0.08
    },
    {
        "Model": "HistGradientBoosting (Baseline)",
        "Algorithm Family": "Gradient Boosting (GBDT)",
        "Key Configuration": "lr=0.10, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2=1.0",
        "CV Macro F1": 0.7487,
        "Student-Test Macro F1": 0.7701,
        "Top-1 Accuracy (%)": 77.17,
        "Route Continuity (%)": 70.29,
        "Log Loss": 0.1221,
        "Inference Latency (ms/obs)": 0.15
    },
    {
        "Model": "HistGradientBoosting (Tuned + HMM) [WINNER]",
        "Algorithm Family": "Bayesian Tuned GBDT + HMM / Viterbi",
        "Key Configuration": "Optuna TPE: lr=0.0517, max_iter=125, max_leaf=61, min_leaf=25, l2=1.65; beta=0.10",
        "CV Macro F1": 0.9215,
        "Student-Test Macro F1": 0.7749,
        "Top-1 Accuracy (%)": 81.84,
        "Route Continuity (%)": 99.87,
        "Log Loss": 0.3389,
        "Inference Latency (ms/obs)": 0.44
    }
]

df_table = pd.DataFrame(records)
df_table.to_csv("results/model_comparison_table.csv", index=False)

# Markdown table
md_content = """# PathSnap Model Comparison - Consolidated Table (Deliverables Section 3.1)

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student-Test Macro F1 | Top-1 Accuracy (%) | Route Continuity (%) | Log Loss | Inference Latency (ms/obs) |
|---|---|---|---|---|---|---|---|---|
"""
for r in records:
    md_content += f"| **{r['Model']}** | {r['Algorithm Family']} | `{r['Key Configuration']}` | {r['CV Macro F1']:.4f} | {r['Student-Test Macro F1']:.4f} | {r['Top-1 Accuracy (%)']:.2f}% | {r['Route Continuity (%)']:.2f}% | {r['Log Loss']:.4f} | {r['Inference Latency (ms/obs)']:.2f} ms |\n"

with open("results/model_comparison_table.md", "w") as f:
    f.write(md_content)

print("[OK] Saved results/model_comparison_table.csv and results/model_comparison_table.md")

# 2. Plots for Section 3.3
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

models_5 = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "HistGradientBoosting"]
cv_f1_5 = [0.5487, 0.6548, 0.6669, 0.7534, 0.7487]
test_top1_5 = [68.21, 65.48, 66.69, 70.43, 77.17]
test_f1_5 = [0.5449, 0.5284, 0.6120, 0.7684, 0.7701]
log_loss_5 = [0.3241, 0.4215, 0.0700, 0.2511, 0.1221]
colors = ['#4575b4', '#74add1', '#abd9e9', '#fdae61', '#d73027']

# Plot 1: Cross-Validation Macro F1 Comparison
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(models_5, cv_f1_5, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Cross-Validation Macro F1-score", fontsize=12, fontweight='bold')
ax.set_title("Cross-Validation Macro F1-Score Across 5 Candidate Classification Models", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 1.0)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., y + 0.02, f"{y:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig("results/visualizations/cv_macro_f1_comparison.png", dpi=300)
plt.close()

# Plot 2: Student-Testing Top-1 Accuracy and Macro F1 Comparison
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(models_5))
width = 0.35
rects1 = ax.bar(x - width/2, test_top1_5, width, label='Top-1 Observation Accuracy (%)', color='#2b83ba', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, [f * 100 for f in test_f1_5], width, label='Macro F1-Score (x100)', color='#d7191c', edgecolor='black', linewidth=0.8)
ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
ax.set_title("Student-Testing Top-1 Accuracy & Macro F1 Across Models", fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models_5, rotation=15, ha='right', fontsize=11)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', frameon=True, fontsize=11)
for rect in rects1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h/100:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("results/visualizations/student_test_top1_macro_f1.png", dpi=300)
plt.close()

# Plot 3: Log Loss Comparison Across Applicable Models
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(models_5, log_loss_5, color=['#313695', '#4575b4', '#74add1', '#fdae61', '#f46d43'], width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Log Loss (Cross-Entropy)", fontsize=12, fontweight='bold')
ax.set_title("Log Loss / Calibration Error Across 5 Candidate Models", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 0.5)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., y + 0.01, f"{y:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig("results/visualizations/log_loss_comparison.png", dpi=300)
plt.close()

# Plot 4: Bayesian Optuna Tuning Progression (Objective vs Trial)
trials_df = pd.read_csv("results/bayesian_optimization_trials.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(trials_df['trial_number'], trials_df['cv_macro_f1'], marker='o', color='#2b83ba', linewidth=2, markersize=7, label='Trial CV Macro F1')
best_cum = trials_df['cv_macro_f1'].cummax()
ax.plot(trials_df['trial_number'], best_cum, linestyle='--', color='#d7191c', linewidth=2.5, label='Best Cumulative Macro F1')
best_trial = trials_df.loc[trials_df['cv_macro_f1'].idxmax()]
ax.scatter([best_trial['trial_number']], [best_trial['cv_macro_f1']], color='gold', s=160, zorder=5, edgecolor='black', label=f"Best Trial #{int(best_trial['trial_number'])}: {best_trial['cv_macro_f1']:.4f}")
ax.set_xlabel("Optuna Trial Number", fontsize=11, fontweight='bold')
ax.set_ylabel("3-Fold Group CV Macro F1", fontsize=11, fontweight='bold')
ax.set_title("Bayesian Hyperparameter Optimization (Optuna TPE) Progression", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0.7, 0.95)
ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig("results/visualizations/bayesian_optuna_progression.png", dpi=300)
plt.close()

# Plot 5: HMM Route Reconstruction Gain
fig, ax = plt.subplots(figsize=(8.5, 5))
categories = ["Route Continuity (%)", "Sequence Route Arc F1", "Normalized Edit Sim"]
classifier_only_vals = [70.29, 59.68, 44.41]
classifier_hmm_vals = [99.87, 81.08, 72.67]
x = np.arange(len(categories))
width = 0.35
r1 = ax.bar(x - width/2, classifier_only_vals, width, label='Classifier Only (Argmax)', color='#fdae61', edgecolor='black')
r2 = ax.bar(x + width/2, classifier_hmm_vals, width, label='Classifier + HMM/Viterbi', color='#2ca25f', edgecolor='black')
ax.set_ylabel("Metric Value (%)", fontsize=12, fontweight='bold')
ax.set_title("Sequential Route Reconstruction: Classifier-Only vs Classifier+HMM", fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(loc='upper left', frameon=True, fontsize=11)
for rect in r1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
for rect in r2:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("results/visualizations/hmm_reconstruction_gains.png", dpi=300)
plt.close()

print("[OK] Generated all 5 comparison visualizations in results/visualizations/")
