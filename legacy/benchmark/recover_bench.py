"""Recoverability and degenerate-feature screens for every generator on every series.
R2_same / R2_lag: in-sample OLS R^2 of y_t on the generator's row t / row t-1.
P_same / P_lag: partial R^2 of the same rows once the admissible lags of y (ending at
t-1) are already in the regression, i.e. the share of the part of y_t that the past cannot
explain which the generator row recovers. Partial R^2 is the informative quantity for
persistent series, where plain R^2 is near one for any level-tracking feature.
Computed over the pre-test portion only. Degenerate: a (near-)constant column or a pair
of columns with |r| > 0.999. Output: results/recover.json"""
import json, numpy as np
from gens import GENS, gen_matrix
from run_bench import CFG
def r2(X, y):
    X = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - np.sum((y - X@b)**2)/np.sum((y - y.mean())**2)
panels = np.load('panels.npy', allow_pickle=True).item(); out = {}
for ds, c in CFG.items():
    for sid, yr in panels[ds].items():
        T = len(yr); t0 = T - c['H']*c['R']; y = yr/yr[:t0].mean(); lo = max(c['lags'])
        for g, (f, frt) in GENS.items():
            for ver in (['full', 'rt'] if frt else ['full']):
                G = gen_matrix(g, 'O*' if ver == 'rt' else 'O', y, t0, c['m'])
                tr = slice(lo, t0); Gt = G[tr]; yt = y[tr]; Gl = G[lo-1:t0-1]
                sd = Gt.std(0); const = bool(np.any(sd < 1e-10))
                coll = False
                if Gt.shape[1] > 1 and not const:
                    C = np.corrcoef(Gt.T); coll = bool(np.any(np.abs(C[np.triu_indices_from(C, 1)]) > 0.999))
                L = np.array(c['lags']); idx = np.arange(lo, t0)
                Lg = y[idx[:, None] - L[None, :]]                     # lag 1 = y[t-1]
                r0 = r2(Lg, yt)
                part = lambda Z: float((r2(np.hstack([Lg, Z]), yt) - r0)/(1 - r0))
                out[f'{ds}|{sid}|{g}|{ver}'] = dict(R2_same=float(r2(Gt, yt)), R2_lag=float(r2(Gl, yt)),
                                                   P_same=part(Gt), P_lag=part(Gl),
                                                   constant=const, collinear=coll)
    print('done', ds, flush=True)
json.dump(out, open('results/recover.json', 'w'))
