# Supplement — anonymised code, protocols and per-series results

Accompanies "When Simulation-Derived Features Leak the Target". Every number in the
manuscript is produced by these scripts and stored in these result files; nothing is
hard-coded in the paper.

## Setup

    pip install numpy pandas scipy scikit-learn xgboost torch statsmodels matplotlib

The M5 competition data are public but not redistributable here. Place them in `./m5_data`
(`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`). The forecasting
pipeline under audit is expected in `./pipeline`.

## Scripts

| File | What it does |
|---|---|
| `xgb_asis.py` | the original (leaky) protocol, as coded |
| `xgb_leakfree.py` | the corrected protocol: origin-dated inputs, targets after the origin |
| `dl_leakfree.py` | corrected protocol, LSTM and Transformer |
| `stat_bench.py` | seasonal naive and ETS benchmarks |
| `metrics_run.py` | per-window errors, for MASE and Diebold-Mariano |
| `recover.py` | recoverability of the target from same-index features |
| `project_test.py` | forward-looking twin design (origin-dated projections, two-stage capping) |
| `corr.py` | correlation structure of the feature sets |
| `wider.py` | corrected protocol at a finer aggregation (store-department) |

Set the horizon with the `H` environment variable, e.g. `H=28 python xgb_leakfree.py`.

## Results files

`leakfree_h{7,14,28}.json`, `dl_leakfree_h{7,14,28}.json`, `asis_h{7,14,28}.json`,
`stat_h{7,14,28}.json`, `metrics_h7.json`, `project_h7.json`, `recoverability.json`,
`final_all_horizons.json`, `scalefree_summary.json`. All are per-series.

## Audit instrument

`AUDIT_PROTOCOL.md` and `audit_coding_sheet.csv` give the pre-specified procedure for
measuring how often published studies state the index at which simulation-derived
predictors are read. This is provided for future work; the manuscript reports a mechanism,
not a prevalence.

## Note

`VERIFICATION_REPORT.md` records the full audit trail, including analyses that did not
enter the manuscript.
