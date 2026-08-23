# UCI Online Retail v0.1 methodology

## Source

The benchmark uses **Online Retail** from the UCI Machine Learning Repository (dataset 352, DOI `10.24432/C5BW33`). The source dataset is CC BY 4.0. Raw data is downloaded at benchmark time and is not committed to this repository.

Pinned source integrity:

- archive SHA256: `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`
- extracted workbook SHA256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`

## Cleaning contract

The raw workbook has 541,909 rows. The warehouse view removes cancellation invoices, non-positive quantities, non-positive prices, missing timestamps, and known service/non-physical stock codes. This leaves 527,794 positive physical-shipment lines, 19,778 invoices, and 3,806 physical SKU codes.

This cleaning policy is intended for outbound warehouse demand and picking analysis. It is not an accounting reconstruction of returns or net revenue.

## Temporal forecasting protocol

The first and last partial calendar weeks are excluded, leaving 51 complete weeks. The top 250 SKUs are chosen using only history before the development period.

- development starts: 2011-08-29
- held-out test starts: 2011-10-10
- held-out test horizon: 8 weeks

Features use only prior observations: lag 1, lag 2, lag 4, four/eight-week means, four-week standard deviation, recent non-zero frequency, and annual week-of-year sine/cosine terms.

Development selection compares:

- last-week naive forecast;
- four-week moving-average forecast;
- two Poisson HistGradientBoosting configurations.

The predeclared ML promotion rule is: candidate WAPE must be no worse than the selected naive baseline, and underforecast units must be no more than 105% of that baseline.

### Measured held-out result

The development-selected baseline is the four-week moving average. The development-selected ML candidate is `hist_gbdt_small`.

| Forecast | Test WAPE | MAE | RMSE | Underforecast units |
|---|---:|---:|---:|---:|
| Last week | 0.6753 | 160.126 | 350.556 | 156,825.00 |
| Four-week moving average | **0.5647** | **133.887** | **284.249** | **139,442.25** |
| HistGradientBoosting | 0.5981 | 141.807 | 332.594 | 174,503.62 |

**ML promotion decision: REJECT.** The nonlinear candidate looked slightly better on development WAPE but did not generalize better than the moving-average baseline on the untouched test period. The rejection is frozen rather than retuning after observing test performance.

## ABC classification

ABC classes are calculated from cumulative positive shipped merchandise value before the test period. The measured pre-test catalogue contains 779 A-class, 959 B-class, and 1,946 C-class physical SKUs.

## Slotting simulation

The source dataset does not contain real warehouse coordinates, dimensions, replenishment constraints, or storage compatibility. Therefore v0.1 uses a deterministic **10-aisle × 25-bay simulated layout** solely to evaluate the optimization method.

Training pick frequency is the number of distinct pre-test invoices containing each selected SKU. Linear assignment minimizes pick-frequency-weighted slot distance. The optimized layout is then evaluated on later, held-out real invoice composition using a return-routing distance proxy.

Measured on 4,307 held-out invoices:

- baseline route proxy: 800,664 distance units;
- optimized route proxy: 718,514 distance units;
- reduction: **10.26%**.

This is evidence that the assignment method improves the defined simulated-layout proxy on later real orders. It is **not** a claim of 10.26% real-world walking-time, labor-cost, or fulfillment-time savings for the original retailer.

## Inventory targets

The repository reports illustrative one-week order-up-to targets using recent demand plus a `z=1.65` variability buffer. The UCI source has no current on-hand inventory, supplier lead time, order constraints, case packs, shelf capacity, or service-level history. Consequently these values are analytical examples, not purchase-order recommendations.

## Release interpretation

The repository-level v0.1 benchmark passes when the model-selection result is explicit under its frozen rule and the slotting optimization does not worsen the held-out route proxy. A rejected ML candidate is therefore a valid benchmark outcome, not a failed release.

## Reproduction

The benchmark environment is pinned in `requirements-ci.txt`. GitHub Actions reruns the complete benchmark and performs an exact diff against `evals/results/v0.1_uci_online_retail.json` before accepting the release evidence.
