from datetime import date, timedelta

import pandas as pd
import pytest

from modules.inventory import InventoryValidationError, add_inventory_intelligence, validate_inventory
from modules.expiry import classify_expiry
from modules.forecasting import estimate_days_cover


def row(**overrides):
    value = {"medicine_id": "M1", "medicine_name": "Example", "category": "Analgesic", "current_stock": 50,
             "average_daily_consumption": 10, "supplier_lead_time": 3, "safety_stock": 10,
             "reorder_quantity": 100, "manufacturing_date": date(2025, 1, 1),
             "expiry_date": date(2027, 1, 1), "supplier_name": "Synthetic Supplier"}
    value.update(overrides)
    return value


def test_reorder_point_cover_and_suggestion():
    result = add_inventory_intelligence(pd.DataFrame([row(current_stock=40)]), today=date(2026, 1, 1)).iloc[0]
    assert result.reorder_point == 40
    assert result.days_until_stockout == 4
    assert result.stockout_risk == "Moderate"
    assert result.suggested_reorder_quantity == 100


def test_critical_risk_when_stock_at_safety_level():
    result = add_inventory_intelligence(pd.DataFrame([row(current_stock=10)]), today=date(2026, 1, 1)).iloc[0]
    assert result.stockout_risk == "Critical"


def test_zero_consumption_is_unknown_cover_and_low_risk():
    result = add_inventory_intelligence(pd.DataFrame([row(average_daily_consumption=0)]), today=date(2026, 1, 1)).iloc[0]
    assert pd.isna(result.days_until_stockout)
    assert result.stockout_risk == "Low"


def test_invalid_negative_and_duplicate_ids_rejected():
    with pytest.raises(InventoryValidationError):
        validate_inventory(pd.DataFrame([row(current_stock=-1)]))
    with pytest.raises(InventoryValidationError):
        validate_inventory(pd.DataFrame([row(), row()]))


def test_cover_and_expiry_helpers():
    assert estimate_days_cover(12, 3) == 4
    assert estimate_days_cover(12, 0) is None
    assert classify_expiry(date(2026, 1, 30), today=date(2026, 1, 1)) == (29, "Critical")
