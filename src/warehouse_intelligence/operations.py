from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Slot:
    slot_id: str
    aisle: int
    bay: int
    distance: float


def abc_classification(frame: pd.DataFrame, *, before: pd.Timestamp) -> pd.DataFrame:
    history = frame[frame["week_start"] < before]
    value = history.groupby("StockCode", as_index=False)["line_value"].sum()
    value = value.sort_values("line_value", ascending=False).reset_index(drop=True)
    total = float(value["line_value"].sum())
    value["value_share"] = value["line_value"] / total if total else 0.0
    value["cumulative_share"] = value["value_share"].cumsum()
    value["abc_class"] = np.select(
        [value["cumulative_share"] <= 0.80, value["cumulative_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    if not value.empty:
        value.loc[0, "abc_class"] = "A"
    return value


def make_slots(*, aisles: int = 10, bays_per_aisle: int = 25) -> list[Slot]:
    slots: list[Slot] = []
    for aisle in range(1, aisles + 1):
        for bay in range(1, bays_per_aisle + 1):
            distance = 4.0 * aisle + float(bay)
            slots.append(Slot(f"A{aisle:02d}-B{bay:02d}", aisle, bay, distance))
    return slots


def _pick_frequency(frame: pd.DataFrame, *, before: pd.Timestamp, skus: list[str]) -> pd.Series:
    history = frame[(frame["week_start"] < before) & frame["StockCode"].isin(skus)]
    return (
        history.groupby("StockCode")["InvoiceNo"]
        .nunique()
        .reindex(skus, fill_value=0)
        .astype(float)
    )


def assign_slots(
    frame: pd.DataFrame,
    *,
    before: pd.Timestamp,
    skus: list[str],
    slots: list[Slot],
) -> tuple[dict[str, Slot], dict[str, Slot]]:
    if len(skus) > len(slots):
        raise ValueError("Not enough slots for selected SKUs")
    selected_slots = sorted(slots, key=lambda slot: (slot.aisle, slot.bay))[: len(skus)]
    baseline = dict(zip(sorted(skus), selected_slots, strict=True))

    frequency = _pick_frequency(frame, before=before, skus=skus)
    weights = frequency.reindex(skus).to_numpy(dtype=float)
    distances = np.array([slot.distance for slot in selected_slots], dtype=float)
    cost = weights[:, None] * distances[None, :]
    row_ind, col_ind = linear_sum_assignment(cost)
    optimized = {
        skus[row]: selected_slots[col] for row, col in zip(row_ind, col_ind, strict=True)
    }
    return baseline, optimized


def route_distance(order_skus: set[str], assignment: dict[str, Slot]) -> float:
    slots = [assignment[sku] for sku in order_skus if sku in assignment]
    if not slots:
        return 0.0
    deepest_by_aisle: dict[int, int] = {}
    for slot in slots:
        deepest_by_aisle[slot.aisle] = max(deepest_by_aisle.get(slot.aisle, 0), slot.bay)
    vertical = 2.0 * sum(deepest_by_aisle.values())
    horizontal = 2.0 * 4.0 * max(deepest_by_aisle)
    return vertical + horizontal


def evaluate_slotting(
    frame: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    test_start: pd.Timestamp,
    skus: list[str],
) -> dict[str, object]:
    slots = make_slots()
    baseline, optimized = assign_slots(frame, before=train_before, skus=skus, slots=slots)
    test = frame[(frame["week_start"] >= test_start) & frame["StockCode"].isin(skus)]
    order_groups = test.groupby("InvoiceNo")["StockCode"].agg(
        lambda values: set(values.astype(str))
    )
    baseline_distances = order_groups.map(lambda items: route_distance(items, baseline))
    optimized_distances = order_groups.map(lambda items: route_distance(items, optimized))
    baseline_total = float(baseline_distances.sum())
    optimized_total = float(optimized_distances.sum())
    reduction = 1.0 - optimized_total / baseline_total if baseline_total else 0.0

    frequency = _pick_frequency(frame, before=train_before, skus=skus)
    top_skus = frequency.sort_values(ascending=False).head(15).index.tolist()
    recommendations = [
        {
            "stock_code": sku,
            "training_order_frequency": int(frequency.loc[sku]),
            "baseline_slot": asdict(baseline[sku]),
            "optimized_slot": asdict(optimized[sku]),
        }
        for sku in top_skus
    ]
    return {
        "objective": "minimize training pick-frequency-weighted slot distance",
        "held_out_route_proxy": "return-routing distance across later observed invoices",
        "orders_evaluated": int(len(order_groups)),
        "baseline_distance": baseline_total,
        "optimized_distance": optimized_total,
        "reduction_fraction": float(reduction),
        "recommendations": recommendations,
    }


def inventory_targets(
    panel: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    limit: int = 20,
    z_value: float = 1.65,
) -> list[dict[str, object]]:
    history = panel[panel["week_start"] <= as_of].copy()
    latest_rows: list[dict[str, object]] = []
    for sku, group in history.groupby("StockCode"):
        tail = group.sort_values("week_start").tail(8)
        if len(tail) < 4:
            continue
        demand = tail["target_units"].to_numpy(dtype=float)
        mean = float(np.mean(demand[-4:]))
        variability = float(np.std(demand, ddof=0))
        target = int(np.ceil(max(0.0, mean + z_value * variability)))
        latest_rows.append(
            {
                "stock_code": str(sku),
                "four_week_mean_units": mean,
                "eight_week_std_units": variability,
                "one_week_order_up_to_target": target,
            }
        )
    latest_rows.sort(key=lambda row: row["one_week_order_up_to_target"], reverse=True)
    return latest_rows[:limit]
