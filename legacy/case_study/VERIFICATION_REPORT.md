# Verification report — DT forecasting paper (ESWA version)

**Purpose.** Check the manuscript's accuracy figures against its own code and the M5 data,
and fill the three unlabelled cells of Figure 15.

**Data.** Full M5 competition data (Kaggle release), obtained from Nixtla's public GitHub
mirror (`github.com/Nixtla/m5-forecasts`). Aggregated to 30 store-category series exactly
as the pipeline does. Test period days 1,760–1,941.

---

## 1. The code leaks the target

Three defects, each sufficient on its own:

1. **Conventional features include the target day.** `build_conventional_features`
   computes `rolling_mean/std/min/max` with `sales.rolling(w)` and no `shift`, so the value
   on day *t* contains `d_t`.
2. **Twin features are recorded after the twin consumes that day's demand.** In
   `DigitalTwinSimulator.simulate`, step 2 applies `d_t`, and only then are
   `inv_position[t]`, `stockout_flag[t]`, `fill_rate_7d[t]` recorded.
3. **The walk-forward loop predicts each test day from features dated that same day**
   (`X_test = feat_arr[w_start:w_end]`). A "7-day forecast" is seven same-day predictions,
   each informed by that day's outcome.

This contradicts manuscript §3.2.5, which states that features are dated at the forecast
origin *t* and targets at *t + h*.

## 2. Accuracy, 7-day horizon, 30 series (mean RMSE, units/day)

| | XGBoost | XGBoost+DT | Twin effect |
|---|---|---|---|
| Manuscript (reported) | 3.91 | 3.24 | −17.1% |
| Pipeline as coded (leaky) | 171.05 | 161.96 | −5.3% |
| **Leak-free, origin-dated features** | **225.57** | **223.41** | **−1.0%** |
| Naive: same as last week | 246.06 | | |

**Leak-free protocol.** For each window the forecast origin *o* is the last observed day.
Training pairs are (features at day *j*, demand at *j + k*) for *k* = 1…7, restricted to
*j + k ≤ o*. One pooled XGBoost per window takes *k* as an input; at prediction time only
the origin-dated feature vector is used. Hyperparameters as in the pipeline.

**Per series (leak-free):** twin features better on 15/30, worse on 15/30; median change
0.0%; range 12.0% better to 7.2% worse. Wilcoxon signed-rank across series, p = 0.49.
Leakage made the as-coded XGBoost look 32% more accurate than the leak-free version.

### Complete leak-free results: 3 horizons x 3 model families

Protocol as above. Deep models refitted quarterly (epoch cap 40, patience 6, batch 128) so
the run fits one CPU core; identical treatment for baseline and +DT models, so the paired
comparison is unaffected. 30 series, bootstrap CI over 10,000 resamples, Wilcoxon
signed-rank across series, Holm correction across all nine tests.

**Positive effect = the twin features made forecasts WORSE.**

| h | model | no twin | + twin | effect | 95% CI | +DT better | p | Holm p |
|---|---|---|---|---|---|---|---|---|
| 7 | XGBoost | 225.6 | 223.4 | −1.0% | [−2.1%, +0.1%] | 15/30 | 0.490 | 0.980 |
| 7 | LSTM | 196.5 | 199.2 | +1.3% | [−0.6%, +3.5%] | 13/30 | 0.164 | 0.730 |
| 7 | Transformer | 203.3 | 208.9 | +2.8% | [+0.2%, +5.5%] | 8/30 | 0.016 | 0.131 |
| 14 | XGBoost | 232.7 | 229.4 | −1.4% | [−2.9%, +0.1%] | 17/30 | 0.146 | 0.730 |
| 14 | LSTM | 217.1 | 220.8 | +1.7% | [−0.0%, +3.3%] | 10/30 | 0.008 | 0.069 |
| 14 | Transformer | 216.3 | 222.3 | +2.8% | [+0.3%, +5.2%] | 11/30 | 0.052 | 0.366 |
| 28 | XGBoost | 237.2 | 237.3 | +0.0% | [−1.2%, +1.4%] | 18/30 | 0.253 | 0.760 |
| 28 | LSTM | 225.5 | 231.6 | +2.7% | [+0.5%, +5.3%] | 11/30 | 0.073 | 0.439 |
| 28 | Transformer | 219.5 | 223.1 | +1.6% | [−1.0%, +4.9%] | 12/30 | 0.503 | 0.980 |

**Conclusions.**
- The manuscript's figures are roughly fifty times smaller than anything this code
  produces; they did not come from it.
- **No effect survives correction for multiple testing.** After Holm adjustment across the
  nine tests, none of the nine is significant. The twin features neither help nor
  reliably hurt: every point estimate lies between −1.4% and +2.8%.
- The direction is nonetheless consistent: for the two deep families the point estimate
  is positive (worse) at all three horizons, six of six.
- **The manuscript's two headline claims are both contradicted.** It reports
  Transformer+DT improving on the Transformer by 18.5%, 20.9% and 22.3% at 7, 14 and 28
  days, growing with horizon. Leak-free the same comparison gives +2.8%, +2.8% and +1.6%
  — the wrong sign, an order of magnitude smaller, and with no growth.
- All families beat a naive last-week forecast, so the forecasting setup works; it is the
  twin layer that adds nothing.

## 2b. Two further measurements (added after the paper draft)

**The leak's value at every horizon** (XGBoost, mean RMSE over 30 series):

| h | leaky base | leaky +twin | apparent | corrected base | corrected +twin | real | error understated |
|---|---|---|---|---|---|---|---|
| 7 | 171.1 | 162.0 | −5.3% | 225.6 | 223.4 | −1.0% | 24% |
| 14 | 176.8 | 168.4 | −4.7% | 232.7 | 229.4 | −1.4% | 24% |
| 28 | 179.3 | 172.9 | −3.6% | 237.2 | 237.3 | +0.0% | 24% |

**Recoverability of the target from same-index features** (OLS, mean in-sample R² for
d_t, 30 series, training period):

| feature set | R² |
|---|---|
| lags 7/14/21/28 (legitimate) | 0.619 |
| rolling statistics, same day | 0.458 |
| twin state, same day | 0.446 |
| twin state, lagged one day | 0.242 |
| all same-day features | 0.736 |

Same-day indexing nearly doubles what the twin state discloses about the target
(0.446 vs 0.242), with no change to the simulator or the split.

## 2c. Can the twin be used in a leak-free, forward-looking way? No.

At the origin the twin legitimately knows stock on hand and the arrival dates of orders
already placed. Two designs were tested on the 7-day horizon, 30 series:

| system | mean RMSE | vs baseline |
|---|---|---|
| A baseline (conventional, origin-dated) | 225.6 | — |
| B + five origin-dated projection features | 227.5 | +0.8%, CI [−0.3, +2.0], better on 10/30, p = 0.100 |
| C two-stage, forecast capped at projected availability | 225.6 | **identical on all 30 series** |

C is null for a structural reason: an (s,S) policy parameterised to a service level keeps
projected availability above projected demand on essentially every day, so the cap never
binds. Availability-aware forecasting needs *observed* availability; a twin calibrated to
a service target simulates precisely the policy that avoids the shortages the method would
exploit.

## 3. Figure 15 (correlations)

Recomputed from the real data under four plausible methods and compared with the 138
labelled values of the published figure:

| Method | mean \|error\| | max \|error\| | sign agreement |
|---|---|---|---|
| pooled across series | 0.39 | 1.02 | 42% |
| per-series, averaged | 0.21 | 0.69 | 56% |
| per-series, standardised | 0.21 | 0.69 | 56% |
| single series (CA_1_FOODS) | 0.24 | 0.83 | 48% |

**The published matrix cannot be reproduced.** Example: inventory position × days of
supply is +0.72 in the figure and −0.30 pooled. The three unlabelled pairs, recomputed
(per-series averaged): sales lag 7 × fill rate −0.29; rolling mean 7d × stockout flag
+0.19; sell price × fill rate −0.02. They are **not** inserted into the figure, because
its other 138 values do not hold.

Two defects in the twin features themselves:
- **`sc_stress` is zero on every day of every series,** training and test. It carries no
  information, and no SHAP or attention result about it can come from this model.
- **`inv_position` and `days_of_supply` correlate at r = 1.00 within each series**
  (one is the other divided by a constant). The eight features carry at most six distinct
  signals, one of them dead.

The paper's qualitative claim of weak twin–demand correlation roughly holds (per-series
mean |r| = 0.08, max 0.43), but it is moot given §2.

## 4. What remains

- **Leak-free LSTM and Transformer** — complete for the 7-day horizon (above).
- **14- and 28-day horizons** — not yet re-run leak-free. Needed before a rewrite, because
  the paper's horizon claim (gains grow with horizon) can only be tested there.

## Files

| File | Content |
|---|---|
| `xgb_asis.py` | pipeline XGBoost exactly as coded |
| `xgb_leakfree.py` | leak-free, origin-dated protocol |
| `corr.py` | correlation recomputation and comparison |
| `dl_leakfree.py` | leak-free LSTM/Transformer |
| `dl_leakfree_h7.json`, `final_h7_summary.json` | deep-learning per-series results and final test table |
| `asis_h7.json`, `leakfree_h7.json`, `summary_h7.json` | per-series results |
| `recomputed_correlations.json` | full recomputed matrix |
