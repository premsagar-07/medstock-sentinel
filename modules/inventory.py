"""Inventory validation and explainable replenishment calculations."""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "medicine_id", "medicine_name", "category", "current_stock",
    "average_daily_consumption", "supplier_lead_time", "safety_stock",
    "reorder_quantity", "manufacturing_date", "expiry_date", "supplier_name",
]
NUMERIC_COLUMNS = ["current_stock", "average_daily_consumption", "supplier_lead_time", "safety_stock", "reorder_quantity"]
DATE_COLUMNS = ["manufacturing_date", "expiry_date"]


class InventoryValidationError(ValueError):
    """Raised when inventory data is incomplete or invalid."""


def validate_inventory(data: pd.DataFrame, *, allow_empty: bool = True) -> pd.DataFrame:
    """Return a normalized copy or raise a useful validation error."""
    if not isinstance(data, pd.DataFrame):
        raise InventoryValidationError("Inventory must be a table.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise InventoryValidationError("Missing required fields: " + ", ".join(missing))
    frame = data[REQUIRED_COLUMNS].copy()
    if not allow_empty and frame.empty:
        raise InventoryValidationError("Add at least one medicine record.")
    for column in ["medicine_id", "medicine_name", "category", "supplier_name"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise InventoryValidationError(f"{column.replace('_', ' ').title()} cannot be blank.")
    if frame["medicine_id"].duplicated().any():
        raise InventoryValidationError("Medicine IDs must be unique.")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise InventoryValidationError(f"{column.replace('_', ' ').title()} must contain finite numbers.")
        if (frame[column] < 0).any():
            raise InventoryValidationError(f"{column.replace('_', ' ').title()} cannot be negative.")
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
        if frame[column].isna().any():
            raise InventoryValidationError(f"{column.replace('_', ' ').title()} must be a valid date.")
    bad_dates = frame["manufacturing_date"] > frame["expiry_date"]
    if bad_dates.any():
        raise InventoryValidationError("Manufacturing date must be on or before expiry date.")
    return frame.reset_index(drop=True)


def load_inventory_csv(source: Any) -> pd.DataFrame:
    """Parse and validate an uploaded inventory CSV."""
    try:
        return validate_inventory(pd.read_csv(source))
    except InventoryValidationError:
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InventoryValidationError(f"Could not read inventory CSV: {type(exc).__name__}.") from exc


def add_inventory_intelligence(data: pd.DataFrame, *, today: date | None = None) -> pd.DataFrame:
    """Calculate days of cover, reorder point, risk, expiry and order suggestion."""
    frame = validate_inventory(data)
    today = today or date.today()
    daily = frame["average_daily_consumption"].replace(0, np.nan)
    frame["days_until_stockout"] = np.floor(frame["current_stock"] / daily).fillna(np.inf)
    frame["reorder_point"] = np.ceil(frame["average_daily_consumption"] * frame["supplier_lead_time"] + frame["safety_stock"])
    frame["stockout_risk"] = np.select(
        [frame["current_stock"] <= frame["safety_stock"], frame["days_until_stockout"] <= frame["supplier_lead_time"], frame["current_stock"] <= frame["reorder_point"]],
        ["Critical", "High", "Moderate"], default="Low",
    )
    expiry_days = frame["expiry_date"].map(lambda value: (value - today).days)
    frame["days_to_expiry"] = expiry_days
    frame["expiry_risk"] = np.select([expiry_days < 0, expiry_days <= 30, expiry_days <= 90], ["Expired", "Critical", "Soon"], default="Low")
    target = frame["reorder_point"] + frame["reorder_quantity"]
    frame["suggested_reorder_quantity"] = np.where(frame["stockout_risk"].isin(["Critical", "High", "Moderate"]), np.maximum(0, target - frame["current_stock"]), 0).astype(int)
    frame["days_until_stockout"] = frame["days_until_stockout"].replace(np.inf, np.nan)
    return frame


def summarize_inventory(data: pd.DataFrame, *, today: date | None = None) -> dict[str, Any]:
    frame = add_inventory_intelligence(data, today=today)
    return {
        "total_medicines": int(len(frame)),
        "total_quantity": int(frame["current_stock"].sum()),
        "low_stock_medicines": int(frame["stockout_risk"].isin(["Moderate", "High", "Critical"]).sum()),
        "critical_alerts": int((frame["stockout_risk"] == "Critical").sum()),
        "expiring_medicines": int(frame["expiry_risk"].isin(["Critical", "Soon", "Expired"]).sum()),
    }
