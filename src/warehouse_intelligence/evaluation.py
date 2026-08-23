from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from .data import (
    UCI_DATASET_ID,
    UCI_DATASET_URL,
    UCI_DOI,
    clean_transactions,
    load_online_retail,
    sha256_file,
)
from .forecasting import build_weekly_features, complete_weeks, fit_and_evaluate, select_skus
from .operations import abc_classification, evaluate_slotting, inventory_targets


def evaluate(
    data_dir: str | Path,
    *,
    download: bool = False,
    seed: int = 42,
    sku_limit: int = 250,
) -> dict[str, object]:
    dataset = load_online_retail(data_dir, download=download)
    frame = clean_transactions(dataset.frame)
    weeks = complete_weeks(frame)
    if len(weeks) < 24:
        raise ValueError("Not enough complete weeks for train/dev/test evaluation")
    test_weeks = 8
    dev_weeks = 6
    test_start = weeks[-test_weeks]
    dev_start = weeks[-(test_weeks + dev_weeks)]

    skus = select_skus(frame, before=dev_start, limit=sku_limit)
    panel = build_weekly_features(frame, skus=skus, weeks=weeks)
    forecast = fit_and_evaluate(panel, dev_start=dev_start, test_start=test_start, seed=seed)
    slotting = evaluate_slotting(
        frame,
        train_before=test_start,
        test_start=test_start,
        skus=skus,
    )
    abc = abc_classification(frame, before=test_start)
    class_counts = abc["abc_class"].value_counts().to_dict()
    targets = inventory_targets(panel, as_of=weeks[-1])

    forecast_promote = forecast["promotion"]["decision"] == "PROMOTE"
    slotting_promote = slotting["optimized_distance"] <= slotting["baseline_distance"]
    return {
        "release": "v0.1",
        "dataset": {
            "name": "Online Retail",
            "uci_dataset_id": UCI_DATASET_ID,
            "doi": UCI_DOI,
            "source_url": UCI_DATASET_URL,
            "license": "CC BY 4.0",
            "raw_rows": len(dataset.frame),
            "clean_physical_shipment_rows": len(frame),
            "invoices": int(frame["InvoiceNo"].nunique()),
            "physical_skus": int(frame["StockCode"].nunique()),
            "date_min": frame["InvoiceDate"].min().isoformat(),
            "date_max": frame["InvoiceDate"].max().isoformat(),
            "archive_sha256": (
                sha256_file(dataset.archive_path) if dataset.archive_path.exists() else None
            ),
            "workbook_sha256": sha256_file(dataset.workbook_path),
        },
        "protocol": {
            "complete_weeks": len(weeks),
            "sku_selection": f"top {len(skus)} SKUs by units before development window",
            "development_start": dev_start.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_weeks": test_weeks,
            "seed": seed,
        },
        "forecasting": forecast,
        "abc": {
            "basis": "cumulative shipped merchandise value before test window",
            "class_counts": {key: int(value) for key, value in class_counts.items()},
        },
        "slotting": slotting,
        "inventory_targets": {
            "assumption": (
                "illustrative one-week order-up-to target; no on-hand inventory or supplier "
                "lead-time data exists in UCI source"
            ),
            "service_factor_z": 1.65,
            "top_targets": targets,
        },
        "release_gate": {
            "decision": "PASS" if forecast_promote and slotting_promote else "PARTIAL",
            "rule": (
                "forecast candidate must pass its promotion rule and optimized slotting must not "
                "worsen held-out route distance"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def write_report(
    output: str | Path,
    data_dir: str | Path,
    *,
    download: bool = False,
    seed: int = 42,
    sku_limit: int = 250,
) -> dict[str, object]:
    report = evaluate(data_dir, download=download, seed=seed, sku_limit=sku_limit)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
