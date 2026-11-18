# PathSnap model artifacts

This folder separates generated artifacts by model.

- `decision_tree/`: Decision Tree model, metadata, validation/student-test metrics, and its group-CV results.
- `hist_gradient_boosting/`: HistGradientBoosting model, metadata, validation/student-test metrics, and its group-CV results.
- `shared/`: route-alignment diagnostics used by the candidate-label preparation pipeline.
- `needs_review/`: `group_cv_results.csv` was provided without a model-specific filename or metadata link, so it is intentionally not assigned to either model.

The `.joblib` files are serialized scikit-learn models. Load them with a compatible scikit-learn version. Do not treat these files as proof of independently verified GPS ground truth: the metadata notes that route-to-GPS labels were inferred by approximate monotonic geometric alignment.
