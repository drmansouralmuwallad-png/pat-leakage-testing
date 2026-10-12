"""RQ3 cost: PAT run time as a function of series length T and number of features p.

A C2-style pipeline (p lags, rolling statistics, a scaler fitted on the training block) is run
on random-walk series. The script reports the mean time of one pipeline execution, of a full
PAT audit (B=5 draws at two audit rows, 11 executions) and of the shift-sensitivity check.

usage: python scalability.py <workdir> <out.json>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, pat_temporal          # noqa: E402
from run_temporal import shift_check                         # noqa: E402

TEMPLATE = '''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('series.csv')
P = {P}
N_TEST = {N_TEST}
y = df['y']
feat = pd.DataFrame(index=df.index)
for k in range(1, P + 1):
    feat['lag_%d' % k] = y.shift(k)
feat['roll_mean'] = y.shift(1).rolling(7).mean()
feat['roll_std'] = y.shift(1).rolling(7).std()
feat = feat.iloc[P + 7:]
target = y.loc[feat.index]
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test)
'''


def main():
    work, out = sys.argv[1:3]
    os.makedirs(work, exist_ok=True)
    res = []
    rng = np.random.default_rng(0)
    for T in (1000, 5000, 20000, 100000):
        y = 100 + np.cumsum(rng.normal(size=T))
        y = np.maximum(y, 0)
        df = pd.DataFrame({'y': y})
        n_test = max(56, T // 10)
        for p in (10, 50, 200):
            path = os.path.join(work, f'scal_T{T}_p{p}.py')
            open(path, 'w').write(TEMPLATE.format(P=p, N_TEST=n_test))
            pipe = ScriptPipeline(path)
            reps = 3
            pipe.run(df)   # warm-up: imports and caches
            t0 = time.perf_counter()
            for _ in range(reps):
                base = pipe.run(df)
            t_run = (time.perf_counter() - t0) / reps
            ts = []
            for r in range(reps):
                rr = pat_temporal(pipe, df, [T - n_test, T - 1], B=5, scale=2.0, seed=r)
                ts.append(rr['seconds'])
                assert not rr['flagged']
            t1 = time.perf_counter()
            shift_check(base, n_test)
            t_shift = time.perf_counter() - t1 + t_run
            rec = dict(T=T, p=p + 2, t_pipeline=t_run, t_pat=float(np.mean(ts)),
                       pat_runs=rr['n_runs'], t_shift=t_shift)
            res.append(rec)
            print(rec, flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
