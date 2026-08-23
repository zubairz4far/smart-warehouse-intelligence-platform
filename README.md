# Smart Warehouse Intelligence Platform

[![CI](https://github.com/zubairz4far/smart-warehouse-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/smart-warehouse-intelligence-platform/actions/workflows/ci.yml)

Real-data warehouse analytics and optimization on the UCI Online Retail transaction history: temporal SKU demand forecasting, ABC analysis, inventory targets, and held-out slotting evaluation.

## Status

**v0.1 — evaluated UCI Online Retail benchmark.**

The benchmark processes 541,909 raw transactions into 527,794 positive physical-shipment lines across 19,778 invoices and 3,806 physical SKU codes. Raw data is downloaded from UCI and verified by SHA256; it is not committed to this repository.

## Headline results

### Demand forecasting: simpler baseline wins

The top 250 SKUs are selected before the development period. Forecast features use prior-week information only. Development data selects the four-week moving average as the naive baseline and `hist_gbdt_small` as the nonlinear candidate; the final eight complete weeks remain untouched until test evaluation.

| Forecast | Test WAPE ↓ | MAE ↓ | RMSE ↓ | Underforecast units ↓ |
|---|---:|---:|---:|---:|
| Last week | 0.6753 | 160.126 | 350.556 | 156,825.00 |
| **Four-week moving average** | **0.5647** | **133.887** | **284.249** | **139,442.25** |
| HistGradientBoosting | 0.5981 | 141.807 | 332.594 | 174,503.62 |

**ML promotion decision: REJECT.** The nonlinear model looked slightly better on development WAPE but generalized worse than the moving-average baseline on the untouched test period. The repository freezes this rejection instead of retuning after test observation.

### Slotting: held-out route proxy improves 10.26%

A deterministic 10-aisle × 25-bay simulated layout is optimized with linear assignment using only pre-test pick frequency. It is then evaluated on **4,307 later real invoices**.

| Layout | Held-out route proxy ↓ |
|---|---:|
| Baseline deterministic assignment | 800,664 |
| **Frequency-optimized assignment** | **718,514** |

**Reduction: 10.26%.** This is a simulated-layout routing proxy using real order composition, not a claim of 10.26% physical labor or fulfillment-time savings in the source retailer's warehouse.

### ABC and inventory intelligence

Pre-test shipped merchandise value produces **779 A**, **959 B**, and **1,946 C** class physical SKUs. The project also emits illustrative one-week order-up-to targets from recent demand and variability. Those targets are explicitly analytical because the source has no on-hand stock, supplier lead times, case packs, capacity, or service-level history.

Machine-readable evidence: [`evals/results/v0.1_uci_online_retail.json`](evals/results/v0.1_uci_online_retail.json).

Full methodology and limitations: [`docs/uci-online-retail-v0.1.md`](docs/uci-online-retail-v0.1.md).

## Temporal evaluation contract

```text
historical complete weeks                 development        untouched test
|--------------------------------------|------ 6 weeks ------|---- 8 weeks ----|
                                                 2011-08-29       2011-10-10
```

Leakage controls:

- first and last partial weeks are excluded;
- SKU selection is frozen before development;
- forecast features contain prior observations only;
- baseline and model configuration selection use development data only;
- slot assignments use only pre-test invoice frequency;
- later invoices evaluate the slot layout.

## Data integrity

Pinned UCI source hashes:

```text
archive   f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b
workbook  43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d
```

The loader fails if either source differs from these measured files.

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
  --output /tmp/v0.1_uci_online_retail.json

diff -u evals/results/v0.1_uci_online_retail.json /tmp/v0.1_uci_online_retail.json
```

CI runs the same pinned Python/scientific environment and rejects any benchmark output that differs from the frozen evidence.

## Data attribution

Daqing Chen, **Online Retail**, UCI Machine Learning Repository, DOI `10.24432/C5BW33`. Dataset license: CC BY 4.0. Repository code: MIT.

## What this release does not claim

v0.1 is an offline, reproducible warehouse-intelligence benchmark. It does not claim access to the original warehouse layout, current inventory, replenishment lead times, worker paths, fulfillment SLAs, or measured monetary savings.
