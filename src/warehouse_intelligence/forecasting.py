from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "mean_4",
    "mean_8",
    "std_4",
    "nonzero_4",
    "week_sin",
    "week_cos",
]


@dataclass(frozen=True)
class ForecastMetrics:
    wape: float
    mae: float
    rmse: float
    bias_units: float
    underforecast_units: float


def complete_weeks(frame: pd.DataFrame) -> list[pd.Timestamp]:
    weeks = sorted(pd.Timestamp(value) for value in frame["week_start"].dropna().unique())
    if len(weeks) < 20:
        raise ValueError("At least 20 observed weeks are required")
    return weeks[1:-1]


def select_skus(
    frame: pd.DataFrame,
    *,
    before: pd.Timestamp,
    limit: int = 250,
) -> list[str]:
    history = frame[frame["week_start"] < before]
    ranked = history.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
    return ranked.head(limit).index.astype(str).tolist()


def build_weekly_features(
    frame: pd.DataFrame,
    *,
    skus: list[str],
    weeks: list[pd.Timestamp],
) -> pd.DataFrame:
    subset = frame[frame["StockCode"].isin(skus) & frame["week_start"].isin(weeks)]
    demand = subset.groupby(["StockCode", "week_start"], as_index=False)["Quantity"].sum()
    grid = pd.MultiIndex.from_product([skus, weeks], names=["StockCode", "week_start"])
    panel = (
        demand.set_index(["StockCode", "week_start"])["Quantity"]
        .reindex(grid, fill_value=0.0)
        .rename("target_units")
        .reset_index()
        .sort_values(["StockCode", "week_start"])
    )
    grouped = panel.groupby("StockCode", group_keys=False)["target_units"]
    panel["lag_1"] = grouped.shift(1)
    panel["lag_2"] = grouped.shift(2)
    panel["lag_4"] = grouped.shift(4)
    shifted = grouped.shift(1)
    panel["mean_4"] = shifted.groupby(panel["StockCode"]).transform(
        lambda values: values.rolling(4, min_periods=4).mean()
    )
    panel["mean_8"] = shifted.groupby(panel["StockCode"]).transform(
        lambda values: values.rolling(8, min_periods=8).mean()
    )
    panel["std_4"] = shifted.groupby(panel["StockCode"]).transform(
        lambda values: values.rolling(4, min_periods=4).std(ddof=0)
    )
    panel["nonzero_4"] = shifted.groupby(panel["StockCode"]).transform(
        lambda values: values.gt(0).rolling(4, min_periods=4).sum()
    )
    week_number = panel["week_start"].dt.isocalendar().week.astype(float)
    panel["week_sin"] = np.sin(2 * np.pi * week_number / 52.0)
    panel["week_cos"] = np.cos(2 * np.pi * week_number / 52.0)
    return panel.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
    error = predicted - actual
    denominator = float(np.sum(actual))
    wape = float(np.sum(np.abs(error)) / denominator) if denominator else 0.0
    return ForecastMetrics(
        wape=wape,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias_units=float(np.sum(error)),
        underforecast_units=float(np.sum(np.clip(-error, 0.0, None))),
    )


def fit_and_evaluate(
    panel: pd.DataFrame,
    *,
    dev_start: pd.Timestamp,
    test_start: pd.Timestamp,
    seed: int = 42,
) -> dict[str, object]:
    train = panel[panel["week_start"] < dev_start]
    dev = panel[(panel["week_start"] >= dev_start) & (panel["week_start"] < test_start)]
    test = panel[panel["week_start"] >= test_start]
    if min(len(train), len(dev), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target_units"].to_numpy(dtype=float)
    x_dev = dev[FEATURE_COLUMNS]
    y_dev = dev["target_units"].to_numpy(dtype=float)
    x_test = test[FEATURE_COLUMNS]
    y_test = test["target_units"].to_numpy(dtype=float)

    baseline_candidates = {
        "last_week": dev["lag_1"].to_numpy(dtype=float),
        "moving_average_4": dev["mean_4"].to_numpy(dtype=float),
    }
    baseline_dev = {name: _metrics(y_dev, values) for name, values in baseline_candidates.items()}
    selected_baseline = min(baseline_dev, key=lambda name: baseline_dev[name].wape)

    candidate_configs = {
        "hist_gbdt_small": dict(max_leaf_nodes=15, l2_regularization=2.0),
        "hist_gbdt_medium": dict(max_leaf_nodes=31, l2_regularization=1.0),
    }
    dev_models: dict[str, tuple[HistGradientBoostingRegressor, ForecastMetrics]] = {}
    for name, config in candidate_configs.items():
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.06,
            max_iter=220,
            min_samples_leaf=30,
            early_stopping=True,
            random_state=seed,
            **config,
        )
        model.fit(x_train, y_train)
        dev_models[name] = (model, _metrics(y_dev, model.predict(x_dev)))
    selected_candidate = min(dev_models, key=lambda name: dev_models[name][1].wape)
    model = dev_models[selected_candidate][0]

    test_baseline_predictions = {
        "last_week": test["lag_1"].to_numpy(dtype=float),
        "moving_average_4": test["mean_4"].to_numpy(dtype=float),
    }
    test_metrics = {
        name: asdict(_metrics(y_test, values)) for name, values in test_baseline_predictions.items()
    }
    candidate_test = _metrics(y_test, model.predict(x_test))
    test_metrics[selected_candidate] = asdict(candidate_test)
    baseline_test = ForecastMetrics(**test_metrics[selected_baseline])

    promote = (
        candidate_test.wape <= baseline_test.wape
        and candidate_test.underforecast_units <= 1.05 * baseline_test.underforecast_units
    )
    rule = (
        "candidate WAPE <= selected naive baseline and underforecast units <= 105% of baseline"
    )
    return {
        "split": {
            "train_rows": len(train),
            "dev_rows": len(dev),
            "test_rows": len(test),
            "dev_start": dev_start.date().isoformat(),
            "test_start": test_start.date().isoformat(),
        },
        "selection": {
            "baseline": selected_baseline,
            "candidate": selected_candidate,
            "dev_wape": {
                **{name: metrics.wape for name, metrics in baseline_dev.items()},
                **{name: metrics.wape for name, (_, metrics) in dev_models.items()},
            },
        },
        "test_metrics": test_metrics,
        "promotion": {
            "decision": "PROMOTE" if promote else "REJECT",
            "rule": rule,
        },
    }
