# Smart Warehouse Intelligence Platform

[![CI](https://github.com/zubairz4far/smart-warehouse-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/smart-warehouse-intelligence-platform/actions/workflows/ci.yml)

Real-data warehouse analytics and optimization on the UCI Online Retail transaction history.

## v0.1 scope

The first release is deliberately different from a generic forecasting notebook. It connects four warehouse decisions under one temporal evaluation contract:

- **SKU demand forecasting** — one-week-ahead weekly demand using lag/rolling/seasonal features;
- **ABC intelligence** — merchandise-value classes calculated only from pre-test history;
- **slotting optimization** — assignment of high-frequency SKUs to lower-travel locations;
- **inventory targets** — illustrative one-week order-up-to targets from recent demand and variability.

The UCI dataset contains timestamped transactions from a UK non-store retailer between December 2010 and December 2011. Raw data is downloaded from UCI at benchmark time and is not committed here.

## Leakage controls

- first and last partial weeks are excluded;
- SKU selection is frozen before the development window;
- demand features contain only prior-week information;
- model/configuration and naive-baseline selection use the development window;
- the last eight complete weeks are held out for forecast evaluation;
- slot assignments use only pre-test pick frequency;
- later observed invoices are used to evaluate the slotting layout.

## Forecast promotion rule

Two naive baselines are compared on development data: last-week demand and four-week moving average. Two HistGradientBoosting/Poisson configurations are also compared on development data. The selected candidate is promoted on the held-out test period only if:

1. candidate WAPE is no worse than the selected naive baseline; and
2. candidate underforecast units are no more than 105% of the selected baseline.

## Slotting evaluation

A 10-aisle × 25-bay deterministic layout is used because the source dataset contains no real warehouse geometry. The optimization objective minimizes **training pick-frequency-weighted slot distance** using linear assignment. The held-out metric is a separate **return-routing distance proxy** over later real invoices.

This is a layout simulation using real order composition, not a claim about physical travel savings in the original retailer's warehouse.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements-ci.txt
pip install -e . --no-deps

ruff check .
pytest -q
warehouse-intelligence benchmark \
  --download \
  --data-dir .cache/online-retail \
  --output evals/results/v0.1_uci_online_retail.json
```

## Data attribution

Daqing Chen, **Online Retail**, UCI Machine Learning Repository, DOI `10.24432/C5BW33`. The dataset is licensed CC BY 4.0. Repository code is MIT licensed.

## Status

**v0.1 benchmark implementation complete; measured CI evidence pending on this pull request.** No result is claimed until the real-data CI run succeeds and its output is frozen into the repository.
