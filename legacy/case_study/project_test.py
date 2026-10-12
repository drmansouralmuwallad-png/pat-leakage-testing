"""
project_test.py — does origin-dated availability projection help?

Idea. At forecast origin o the twin knows two things that are legitimately about the
future and involve no future demand: the stock on hand at o, and the replenishment orders
already placed whose arrival dates fall after o. From those we can project, for each
target day o+k, how much stock will be available -- without touching any demand after o.

Three systems, all sharing the same leak-free protocol (origin-dated inputs, targets
strictly after the origin, pooled direct multi-step with k as a feature):

  A  baseline        conventional features only                     (= the earlier run)
  B  + projection    baseline plus five origin-dated projection features per step k
  C  two-stage       A's forecast, capped at projected availability for that day

Projection features at step k, all computable at o:
  on_hand_o, cumulative committed arrivals by o+k, days to next arrival,
  projected inventory = on_hand_o + arrivals_k - k*dbar_train,
  projected shortfall indicator.

The cap in C uses A's own forecast to draw down stock, so no observed future demand
enters. Paired Wilcoxon tests across series.
"""
import sys, os, json, time
sys.path.insert(0, './pipeline')
import numpy as np, pandas as pd, xgboost as xgb
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features

H = int(os.environ.get('H', '7'))
OUT = f'./project_h{H}.json'
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
cfg = R.ExperimentConfig()


def simulate_with_schedule(demand, train_end_idx):
    """Re-implements the twin's daily loop, additionally recording on-hand by day and the
    (order_day, arrival_day, qty) schedule, which the packaged simulator does not expose."""
    sim = DigitalTwinSimulator()
    T = len(demand)
    s, S = sim._compute_policy_params(demand[:train_end_idx + 1], 10.0)
    on_hand = S
    pend = []                       # (arrival_day, qty, order_day)
    OH = np.zeros(T); sched = []
    rng = np.random.default_rng(cfg.seed)
    for t in range(T):
        arr = [o for o in pend if o[0] <= t]; pend = [o for o in pend if o[0] > t]
        for a in arr: on_hand += a[1]
        on_hand = max(0.0, on_hand - demand[t])
        on_order = sum(o[1] for o in pend)
        if on_hand + on_order <= s:
            q = S - (on_hand + on_order)
            L = int(rng.integers(sim.config.lead_time_min, sim.config.lead_time_max + 1))
            pend.append((t + L, q, t)); sched.append((t, t + L, q))
        OH[t] = on_hand
    return OH, sched, s, S


def projection_features(o, k_arr, OH, sched, dbar):
    """Everything here is known at day o."""
    fut = [(a, q) for (od, a, q) in sched if od <= o and a > o]
    rows = []
    for k in k_arr:
        cum = sum(q for a, q in fut if a <= o + k)
        nxt = min([a - o for a, q in fut if a > o], default=99)
        proj = OH[o] + cum - k * dbar
        rows.append([OH[o], cum, nxt, proj, 1.0 if proj <= 0 else 0.0])
    return np.array(rows)


def pooled(F, d, o, extra=None):
    J = np.arange(28, o - H + 1)
    X = np.vstack([np.column_stack(
        [F[J], np.full(len(J), k)] + ([extra(J, k)] if extra else [])) for k in range(1, H + 1)])
    y = np.concatenate([d[J + k] for k in range(1, H + 1)])
    return X, y


def fit(X, y):
    vs = int(len(X) * 0.9)
    idx = np.random.default_rng(cfg.seed).permutation(len(X))
    tr, vl = idx[:vs], idx[vs:]
    m = xgb.XGBRegressor(n_estimators=cfg.xgb_n_rounds, max_depth=cfg.xgb_max_depth,
                         learning_rate=cfg.xgb_lr, subsample=cfg.xgb_subsample,
                         colsample_bytree=cfg.xgb_colsample,
                         early_stopping_rounds=cfg.xgb_early_stop,
                         random_state=cfg.seed, n_jobs=1, verbosity=0)
    m.fit(X[tr], y[tr], eval_set=[(X[vl], y[vl])], verbose=False)
    return m


sales, _, _ = R.load_m5_data('./m5_data')
agg, ids = R.aggregate_to_store_category(sales)
for sid in ids:
    if sid in done: continue
    t0 = time.time()
    d = agg.loc[sid].values.astype(float)
    conv = build_conventional_features(pd.Series(d)).fillna(0).values
    OH, sched, s_lvl, S_lvl = simulate_with_schedule(d, cfg.train_end_day - 1)
    dbar = d[:cfg.train_end_day].mean()
    res = {'A_baseline': [], 'B_projection': [], 'C_two_stage': []}
    for w in range((cfg.test_end_day - cfg.val_end_day) // H):
        ws = cfg.val_end_day + w * H; o = ws - 1; y = d[ws:ws + H]
        ks = np.arange(1, H + 1)

        Xa, ya = pooled(conv, d, o); ma = fit(Xa, ya)
        qa = np.column_stack([np.repeat(conv[o][None, :], H, 0), ks])
        pa = ma.predict(qa)

        # B: projection features, computed at every training origin and at o
        def extra(J, k):
            return np.vstack([projection_features(int(j), [k], OH, sched, dbar)[0] for j in J])
        Xb, yb = pooled(conv, d, o, extra=extra); mb = fit(Xb, yb)
        qb = np.hstack([qa, projection_features(o, ks, OH, sched, dbar)])
        pb = mb.predict(qb)

        # C: cap A's forecast at stock projected to be available that day
        avail = OH[o] + np.array([sum(q for (od, a, q) in sched if od <= o and a <= o + k)
                                  for k in ks])
        pc = np.minimum(pa, np.maximum(0.0, avail - np.concatenate([[0], np.cumsum(pa)[:-1]])))

        for nm, p in (('A_baseline', pa), ('B_projection', pb), ('C_two_stage', pc)):
            res[nm].append(float(np.sqrt(np.mean((y - p) ** 2))))
    done[sid] = {k: float(np.mean(v)) for k, v in res.items()}
    json.dump(done, open(OUT, 'w'))
    r = done[sid]
    print(f"{sid:16s} A {r['A_baseline']:7.1f}  B {r['B_projection']:7.1f}  C {r['C_two_stage']:7.1f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
