"""Non-forecasting generalisation experiment: a tabular classification task with target
encoding of a high-cardinality categorical feature (random train/test split, no time).
Task: predict whether an item sells in a store on a given day (M5, last 365 days), from
target-encoded item-store identity, target-encoded department-store, frequency encoding and
day of week. Prediction-time information excludes all test labels. Pipelines:
  TE-train   encodings fitted on training labels only                (admissible)
  TE-full    encodings fitted on all labels, test included           (inadmissible)
  TE-LOO     leave-one-out encodings on all labels: own label excluded,
             other test labels included                              (inadmissible)
  FREQ       frequency encoding only, no labels                      (admissible)
PAT: test labels are redrawn from Bernoulli(training base rate); the feature pipeline is
rerun; any change of a test-row input flags the pipeline. 10 random splits, B=5.
Output: results_te.json"""
import json, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
rng0 = np.random.default_rng(2026)
cols = pd.read_csv('../../data/sales_train_evaluation.csv', nrows=0).columns
days = [c for c in cols if c.startswith('d_')][-365:]
df = pd.read_csv('../../data/sales_train_evaluation.csv', usecols=['item_id', 'dept_id', 'store_id'] + days)
N = 200_000
r = rng0.integers(0, len(df), N); d = rng0.integers(0, 365, N)
vals = df[days].to_numpy()
data = pd.DataFrame({'is': (df['item_id'] + '|' + df['store_id']).to_numpy()[r],
                     'ds': (df['dept_id'] + '|' + df['store_id']).to_numpy()[r],
                     'dow': d % 7, 'y': (vals[r, d] > 0).astype(int)})
data['is'] = pd.factorize(data['is'])[0]; data['ds'] = pd.factorize(data['ds'])[0]
del df, vals
A = 20.0  # smoothing strength of the encodings

def te(cat, y, fit_mask, prior):
    s = pd.Series(y[fit_mask]).groupby(cat[fit_mask]).agg(['sum', 'count'])
    S = np.zeros(cat.max() + 1); C = np.zeros(cat.max() + 1)
    S[s.index] = s['sum']; C[s.index] = s['count']
    return S, C

def features(y, tr, mode):
    """y: labels for all rows (test entries are whatever the pipeline is given)."""
    prior = y[tr].mean()
    X = []
    for c in ('is', 'ds'):
        cat = data[c].to_numpy()
        if mode == 'FREQ':
            X.append(np.bincount(cat)[cat].astype(float)); continue
        fit = tr if mode == 'TE-train' else np.ones(len(y), bool)
        S, C = te(cat, y, fit, prior)
        if mode == 'TE-LOO':
            s_, c_ = S[cat] - y, C[cat] - 1
        else:
            s_, c_ = S[cat], C[cat]
        X.append((s_ + A * prior) / (c_ + A))
    X.append(data['dow'].to_numpy().astype(float))
    return np.column_stack(X)

def fit_auc(learner, Xtr, ytr, Xte, yte):
    if learner == 'logit':
        sc = StandardScaler().fit(Xtr)
        p = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
    else:
        p = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=0,
                               n_jobs=1, verbose=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)

y = data['y'].to_numpy(); out = {'n_rows': N, 'base_rate': float(y.mean()), 'splits': []}
MODES = ['TE-train', 'TE-full', 'TE-LOO', 'FREQ']
for s in range(10):
    rng = np.random.default_rng(100 + s)
    tr = rng.random(N) < 0.8; te_ = ~tr
    rec = {}
    for m in MODES:
        X = features(y, tr, m)
        rec[m] = {L: float(fit_auc(L, X[tr], y[tr], X[te_], y[te_])) for L in ('logit', 'lgbm')}
        # PAT: redraw test labels, rerun pipeline, compare test-row inputs
        t0 = time.perf_counter(); hits = []
        for b in range(5):
            yp = y.copy(); yp[te_] = (rng.random(te_.sum()) < y[tr].mean()).astype(int)
            Xp = features(yp, tr, m)
            hits.append(bool(np.any(np.abs(Xp[te_] - X[te_]) > 1e-9 * (1 + np.abs(X[te_])))))
        rec[m]['pat_hits'] = hits; rec[m]['pat_sec'] = (time.perf_counter() - t0) / 5
    out['splits'].append(rec); print(s, {m: (round(rec[m]['logit'], 4), round(rec[m]['lgbm'], 4), sum(rec[m]['pat_hits'])) for m in MODES}, flush=True)
json.dump(out, open('results_te.json', 'w'), indent=1)
