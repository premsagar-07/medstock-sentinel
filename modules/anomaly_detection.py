"""Prototype unsupervised anomaly detection for synthetic medicine inventory."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_NAME = "IsolationForest Inventory Anomaly Detector"
ALGORITHM = "Scikit-learn IsolationForest (unsupervised)"
MIN_TRAINING_RECORDS = 5
FEATURE_COLUMNS = [
    "current_stock",
    "average_daily_consumption",
    "supplier_lead_time",
    "safety_stock",
    "days_of_stock_cover",
]


def inventory_anomaly_results(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fit a prototype model and return per-medicine labels, scores and explanations.

    The anomaly score is ``-IsolationForest.score_samples``; larger values are
    more unusual relative to this training inventory. It is not an accuracy or
    probability estimate.
    """
    required = [
        "medicine_id", "medicine_name", "current_stock",
        "average_daily_consumption", "supplier_lead_time", "safety_stock",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing inventory features: " + ", ".join(missing))

    frame = data[required].copy()
    for column in required[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required[2:]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["medicine_id", "medicine_name", "status", "anomaly_score", "explanation"]), 0

    daily = frame["average_daily_consumption"]
    cover = frame["current_stock"].div(daily.where(daily > 0))
    # Zero consumption means effectively no consumption-based stockout horizon.
    finite_cover = cover[np.isfinite(cover)]
    cover_cap = max(3650.0, float(finite_cover.max()) if not finite_cover.empty else 3650.0)
    frame["days_of_stock_cover"] = cover.replace([np.inf, -np.inf], np.nan).fillna(cover_cap).clip(upper=cover_cap)

    if len(frame) < MIN_TRAINING_RECORDS:
        return pd.DataFrame(columns=["medicine_id", "medicine_name", "status", "anomaly_score", "explanation"]), len(frame)

    features = frame[FEATURE_COLUMNS].astype(float)
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    labels = model.fit_predict(features)
    scores = -model.score_samples(features)

    medians = features.median()
    results = frame[["medicine_id", "medicine_name"]].copy()
    results["status"] = np.where(labels == -1, "Anomalous", "Normal")
    results["anomaly_score"] = scores.round(4)

    def explain(row: pd.Series, status: str) -> str:
        deviations = (features.loc[row.name] - medians).abs() / (features.median().abs() + 1.0)
        feature = str(deviations.idxmax()).replace("_", " ")
        value = features.loc[row.name, deviations.idxmax()]
        label = "unusual" if status == "Anomalous" else "within the model's learned pattern"
        return f"{feature.title()} is {value:g}; this record is {label} compared with this inventory."

    results["explanation"] = [explain(features.loc[index], results.loc[index, "status"]) for index in results.index]
    return results.sort_values(["status", "anomaly_score"], ascending=[True, False]).reset_index(drop=True), len(frame)
