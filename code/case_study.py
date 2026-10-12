"""RQ5 case study: detect, diagnose, correct and revalidate a digital-twin feature pipeline.

The pipeline is a re-implementation of the author's earlier digital-twin demand-forecasting
feature layer. It contains the three defects documented in its verification report:
rolling statistics computed without a shift, twin state recorded after the day's demand has
been served, and same-day features used for each prediction. v0 is the pipeline as built.
v1 fixes the rolling statistics. v2 also reads the twin state at the previous day. v2u is v2
with an unseeded simulator.

usage: python case_study.py <panels.npy> <out.json>
"""
import json, os, sys, time, zlib
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, pat_temporal, determinism_check

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sources', 'case')
VERSIONS = ['v0_as_built', 'v1_rolling_fixed', 'v2_twin_fixed', 'v2u_unseeded']


def rmse_models(out, use_twin):
    Xtr, _, cols = out['X_train']; Xte = out['X_test'][0]
    ytr = out['y_train'][0].ravel(); yte = out['y_test'][0].ravel()
    keep = [i for i, c in enumerate(cols) if use_twin or not c.startswith('dt_')]
    Xtr, Xte = np.nan_to_num(Xtr[:, keep]), np.nan_to_num(Xte[:, keep])
    res = {}
    r = Ridge(alpha=1.0).fit(Xtr, ytr); res['ridge'] = float(np.sqrt(np.mean((r.predict(Xte) - yte) ** 2)))
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0,
                          verbose=-1, deterministic=True, force_row_wise=True, n_jobs=1).fit(Xtr, ytr)
    res['lgbm'] = float(np.sqrt(np.mean((g.predict(Xte) - yte) ** 2)))
    return res


def main():
    P = np.load(sys.argv[1], allow_pickle=True).item()['M5']
    out = {}
    for sid, y in P.items():
        df = pd.DataFrame({'y': np.asarray(y, float)})
        n = len(df)
        rec = {}
        for v in VERSIONS:
            pipe = ScriptPipeline(os.path.join(ROOT, v + '.py'))
            t0 = time.perf_counter()
            nondet = determinism_check(pipe, df)
            t_det = time.perf_counter() - t0
            base = pipe.run(df)
            res = pat_temporal(pipe, df, [n - 56, n - 1], B=5, scale=2.0,
                               seed=zlib.crc32(f'{sid}|{v}'.encode()), base=base)
            flagged = sorted({c.split(':', 1)[1] for a in res['audits'] for c in a['flagged']})
            r = dict(nondeterministic_cols=nondet, t_determinism=t_det, pat_flagged=res['flagged'],
                     pat_cols=flagged, pat_seconds=res['seconds'], pat_runs=res['n_runs'])
            if v != 'v2u_unseeded':
                r['rmse_with_twin'] = rmse_models(base, True)
                r['rmse_without_twin'] = rmse_models(base, False)
            rec[v] = r
        out[sid] = rec
        print(sid, {v: (rec[v]['pat_flagged'], len(rec[v]['pat_cols']), len(rec[v]['nondeterministic_cols'])) for v in VERSIONS}, flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
