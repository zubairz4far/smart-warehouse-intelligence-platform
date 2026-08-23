import numpy as np
import pandas as pd

from warehouse_intelligence.data import clean_transactions
from warehouse_intelligence.forecasting import FEATURE_COLUMNS, build_weekly_features
from warehouse_intelligence.operations import (
    abc_classification,
    assign_slots,
    make_slots,
    route_distance,
)


def test_clean_transactions_keeps_positive_physical_shipments_only():
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["100", "C101", "102", "103", "104"],
            "StockCode": ["SKU1", "SKU1", "POST", "SKU2", "SKU3"],
            "Description": ["a", "a", "postage", "b", "c"],
            "Quantity": [2, -1, 1, 0, 3],
            "InvoiceDate": pd.to_datetime(["2011-01-01"] * 5),
            "UnitPrice": [2.0, 2.0, 1.0, 4.0, 5.0],
            "CustomerID": [1, 1, 1, 2, 3],
            "Country": ["UK"] * 5,
        }
    )
    cleaned = clean_transactions(frame)
    assert cleaned["StockCode"].tolist() == ["SKU1", "SKU3"]
    assert cleaned["line_value"].tolist() == [4.0, 15.0]
    assert cleaned["week_start"].notna().all()


def test_weekly_features_use_only_prior_demand():
    weeks = list(pd.date_range("2011-01-03", periods=12, freq="7D"))
    frame = pd.DataFrame(
        {
            "StockCode": ["A"] * 12,
            "week_start": weeks,
            "Quantity": np.arange(1, 13, dtype=float),
        }
    )
    panel = build_weekly_features(frame, skus=["A"], weeks=weeks)
    row = panel.iloc[0]
    assert row["target_units"] == 9.0
    assert row["lag_1"] == 8.0
    assert row["mean_4"] == 6.5
    assert set(FEATURE_COLUMNS).issubset(panel.columns)


def test_slotting_moves_high_frequency_sku_closer():
    rows = []
    for invoice in range(20):
        rows.append(
            {
                "InvoiceNo": f"H{invoice}",
                "StockCode": "HIGH",
                "week_start": pd.Timestamp("2011-01-03"),
            }
        )
    for invoice in range(2):
        rows.append(
            {
                "InvoiceNo": f"L{invoice}",
                "StockCode": "LOW",
                "week_start": pd.Timestamp("2011-01-03"),
            }
        )
    frame = pd.DataFrame(rows)
    baseline, optimized = assign_slots(
        frame,
        before=pd.Timestamp("2011-02-01"),
        skus=["LOW", "HIGH"],
        slots=make_slots(aisles=1, bays_per_aisle=2),
    )
    assert optimized["HIGH"].distance <= optimized["LOW"].distance
    assert route_distance({"HIGH"}, optimized) <= route_distance({"HIGH"}, baseline)


def test_abc_classification_marks_top_value_item_a():
    frame = pd.DataFrame(
        {
            "StockCode": ["A", "B", "C"],
            "line_value": [80.0, 15.0, 5.0],
            "week_start": pd.to_datetime(["2011-01-03"] * 3),
        }
    )
    result = abc_classification(frame, before=pd.Timestamp("2011-02-01"))
    assert result.iloc[0]["StockCode"] == "A"
    assert result.iloc[0]["abc_class"] == "A"
