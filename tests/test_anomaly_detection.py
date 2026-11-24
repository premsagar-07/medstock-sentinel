import pandas as pd

from modules.anomaly_detection import (
    FEATURE_COLUMNS,
    inventory_anomaly_results,
)


def test_isolation_forest_trains_and_returns_explainable_inventory_results():
    inventory = pd.DataFrame({
        "medicine_id": [f"MED{i:03}" for i in range(1, 9)],
        "medicine_name": [f"Medicine {i}" for i in range(1, 9)],
        "current_stock": [100, 110, 95, 105, 98, 102, 90, 10000],
        "average_daily_consumption": [10, 11, 9, 10, 10, 10, 8, 1],
        "supplier_lead_time": [5, 5, 6, 5, 5, 4, 5, 90],
        "safety_stock": [20, 20, 18, 20, 20, 20, 15, 500],
    })

    results, training_count = inventory_anomaly_results(inventory)

    assert training_count == 8
    assert len(results) == 8
    assert set(results["status"]) <= {"Normal", "Anomalous"}
    assert pd.api.types.is_numeric_dtype(results["anomaly_score"])
    assert results["explanation"].str.len().gt(0).all()
    assert "MED008" in set(results.loc[results["status"] == "Anomalous", "medicine_id"])
    assert "days_of_stock_cover" in FEATURE_COLUMNS


def test_small_inventory_returns_no_fabricated_predictions():
    inventory = pd.DataFrame({
        "medicine_id": ["MED001", "MED002"],
        "medicine_name": ["A", "B"],
        "current_stock": [10, 20],
        "average_daily_consumption": [1, 2],
        "supplier_lead_time": [3, 4],
        "safety_stock": [2, 4],
    })

    results, training_count = inventory_anomaly_results(inventory)

    assert training_count == 2
    assert results.empty
