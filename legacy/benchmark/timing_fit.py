"""Cost of one model fit (LightGBM, base inputs) at one origin, for comparison with the cost
of one feature-pipeline execution. Five series per panel. Output: results/timing_fit.json"""
import json, time, numpy as np
from run_bench import CFG, base_X, fit_predict
panels = np.load('panels.npy', allow_pickle=True).item(); out = {}
for ds, c in CFG.items():
    ts = []
    for sid, y0 in list(panels[ds].items())[:5]:
        T = len(y0); o = T-1-c['H']; t0 = T-c['H']*c['R']; y = y0/y0[:t0].mean()
        J = np.arange(max(max(c['lags']), o-c['H']-c['W']+1), o-c['H']+1, c['stride']); ks = np.arange(1, c['H']+1)
        Xtr = np.vstack([base_X(y, J, k, c) for k in ks]); ytr = np.concatenate([y[J+k] for k in ks])
        Xte = np.vstack([base_X(y, np.array([o]), k, c) for k in ks])
        t = time.perf_counter(); fit_predict('lgbm', Xtr, ytr, Xte); ts.append(time.perf_counter()-t)
    out[ds] = dict(lgbm_fit_sec=float(np.mean(ts)), rows=int(len(Xtr)))
json.dump(out, open('results/timing_fit.json', 'w'), indent=1); print(out)
