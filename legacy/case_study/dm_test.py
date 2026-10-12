"""Diebold-Mariano tests at h=7 for the case study, from the per-window errors stored in
metrics_h7.json. Squared-error differentials are pooled across the 30 series; long-run
variance by Newey-West (Bartlett weights, lag h-1); Harvey-Leybourne-Newbold small-sample
correction; t reference distribution. Output: dm_h7.json"""
import json, numpy as np
from scipy import stats
d = json.load(open('metrics_h7.json')); H = 7; out = {}
def dm(a, b):
    dd = np.concatenate([np.concatenate([np.array(w) for w in v['errors'][a]])**2 -
                         np.concatenate([np.array(w) for w in v['errors'][b]])**2 for v in d.values()])
    n = len(dd); x = dd - dd.mean()
    g = [np.sum(x[k:]*x[:n-k])/n for k in range(H)]
    lrv = g[0] + 2*sum((1-k/H)*g[k] for k in range(1, H))
    st = dd.mean()/np.sqrt(lrv/n)*np.sqrt((n+1-2*H+H*(H-1)/n)/n)
    return float(st), float(2*stats.t.sf(abs(st), n-1)), n
for a, b in [('SeasonalNaive', 'ETS'), ('ETS', 'XGBoost'), ('ETS', 'XGBoost+DT'), ('XGBoost', 'XGBoost+DT')]:
    s, p, n = dm(a, b); out[f'{a} vs {b}'] = dict(stat=s, p=p, n=n); print(f'{a:14s} vs {b:11s} DM {s:+.2f}  p {p:.3f}  n {n}')
json.dump(out, open('dm_h7.json', 'w'), indent=1)
