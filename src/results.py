import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def classification_metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def regression_metrics(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def save_results(lab: str, results: dict) -> Path:
    """results = {"classification": {setup: metrics}, "regression": {setup: metrics}}"""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{lab}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return path
