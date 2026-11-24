"""Optional explainable demand trend helper (not medical advice)."""
import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_days_cover(stock: float, average_daily_consumption: float) -> float | None:
    if stock < 0 or average_daily_consumption < 0:
        raise ValueError("Stock and consumption must be non-negative.")
    return None if average_daily_consumption == 0 else float(stock / average_daily_consumption)


def fit_consumption_trend(daily_usage: list[float]) -> dict[str, float | str]:
    """Fit a simple linear trend to observed daily usage; minimum 3 points."""
    values = np.asarray(daily_usage, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Provide at least three finite, non-negative daily usage values.")
    model = LinearRegression().fit(np.arange(len(values)).reshape(-1, 1), values)
    return {"daily_change": float(model.coef_[0]), "latest_smoothed_rate": float(max(0, model.predict([[len(values)]])[0])), "method": "linear regression over supplied daily usage"}
