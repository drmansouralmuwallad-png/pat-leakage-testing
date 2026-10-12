"""Controlled benchmark: six target-driven generators x forecast-origin protocols x three
datasets x two learners. For every cell the augmented model is compared with the same
learner on the base (lag + calendar) features only, and with a copy whose generated
inputs are read one step earlier (the shift diagnostic). Results per series go to
results/<dataset>_<learner>.json. Usage: python run_bench.py DATASET LEARNER"""
import sys, os, json, time, warnings
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from gens import CELLS, GENS, gen_matrix, gen_index
warnings.filterwarnings('ignore')

CFG = {  # m: season, H: horizon, R: test origins, W: training origins, stride, lags
 'M5':  dict(m=7,  H=7,  R=8, W=730, stride=1, lags=[1,2,3,4,5,6,7,14,21,28]),
 'M4D': dict(m=7,  H=14, R=6, W=730, stride=1, lags=[1,2,3,4,5,6,7,14,21,28]),
 'M4H': dict(m=24, H=48, R=4, W=600, stride=2, lags=list(range(1,25))+[48,72,168]),
 'ELEC': dict(m=24, H=48, R=4, W=600, stride=2, lags=list(range(1,25))+[48,72,168]),
}

def base_X(y, J, k, c):
    L = np.array(c['lags'])
    lagv = y[J[:, None] - L[None, :] + 1]                 # lag 1 = y[j]
    ph = 2*np.pi*((J + k) % c['m'])/c['m']
    cols = [lagv, np.full(len(J), k), np.sin(ph), np.cos(ph)]
    if c['H'] > c['m']:          # seasonal anchor: value at the target's phase in the last observed cycle
        cols.append(y[J - c['m'] + 1 + (k - 1) % c['m']])
    return np.column_stack(cols)

def fit_predict(learner, Xtr, ytr, Xte, seed=0):
    if learner == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, tree_method='hist', random_state=seed,
                                n_jobs=1).fit(Xtr, ytr).predict(Xte)
    if learner == 'mlp':
        from sklearn.neural_network import MLPRegressor
        if len(Xtr) > 2000:          # fixed-seed subsample of training pairs bounds the cost
            ix = np.random.default_rng(seed).choice(len(Xtr), 2000, replace=False); Xtr, ytr = Xtr[ix], ytr[ix]
        sc = StandardScaler().fit(Xtr)
        m = MLPRegressor(hidden_layer_sizes=(32, 16), learning_rate_init=1e-3, batch_size=256,
                         max_iter=40, early_stopping=True, n_iter_no_change=6, random_state=seed)
        return m.fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
    if learner == 'ridge':
        sc = StandardScaler().fit(Xtr)
        return Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
    return m.fit(Xtr, ytr).predict(Xte)

def run_series(y_raw, c, learner):
    T = len(y_raw); H, R = c['H'], c['R']
    origins = [T-1-H*(R-r) for r in range(R)]
    t0 = origins[0] + 1
    y = y_raw / y_raw[:t0].mean()
    G = {}
    for gname, proto, _ in CELLS:
        key = (gname, 'rt' if proto == 'O*' else 'full')
        if key not in G: G[key] = gen_matrix(gname, proto, y, t0, c['m'])
    maxlag = max(c['lags'])
    variants = ['base'] + [f'{g}|{p}' for g, p, _ in CELLS] + [f'{g}|{p}|s1' for g, p, _ in CELLS]
    err = {v: [] for v in variants}; err['snaive'] = []
    for o in origins:
        J = np.arange(max(maxlag, o-H-c['W']+1), o-H+1, c['stride'])
        ks = np.arange(1, H+1)
        Xb_tr = np.vstack([base_X(y, J, k, c) for k in ks])
        y_tr = np.concatenate([y[J+k] for k in ks])
        Jo = np.array([o])
        Xb_te = np.vstack([base_X(y, Jo, k, c) for k in ks])
        y_te = y[o+1:o+H+1]
        err['snaive'].append(y_te - np.resize(y[o-c['m']+1:o+1], H))
        err['base'].append(y_te - fit_predict(learner, Xb_tr, y_tr, Xb_te))
        for gname, proto, _ in CELLS:
            Gm = G[(gname, 'rt' if proto == 'O*' else 'full')]
            for sh in ((0,) if os.environ.get('NOSHIFT') else (0, 1)):
                Xa_tr = np.hstack([Xb_tr, np.vstack([Gm[gen_index(proto, J, k, sh)] for k in ks])])
                Xa_te = np.hstack([Xb_te, np.vstack([Gm[gen_index(proto, Jo, k, sh)] for k in ks])])
                v = f'{gname}|{proto}' + ('|s1' if sh else '')
                err[v].append(y_te - fit_predict(learner, Xa_tr, y_tr, Xa_te))
    return {v: float(np.sqrt(np.mean(np.concatenate(e)**2))) for v, e in err.items() if e}

if __name__ == '__main__':
    ds, learner = sys.argv[1], sys.argv[2]
    panels = np.load('panels.npy', allow_pickle=True).item()
    os.makedirs('results', exist_ok=True); out = f'results/{ds}_{learner}.json'
    done = json.load(open(out)) if os.path.exists(out) else {}
    lim = int(os.environ.get('LIMIT', '0')) or None
    items = list(panels[ds].items())[:lim]
    if os.environ.get('SUBSET'):   # fixed-seed random subset of series (used for the MLP)
        k = int(os.environ['SUBSET']); pick = set(np.random.default_rng(99).choice(len(panels[ds]), k, replace=False))
        items = [it for i, it in enumerate(panels[ds].items()) if i in pick]
    for sid, y in items:
        if sid in done: continue
        t = time.time(); done[sid] = run_series(y, CFG[ds], learner)
        json.dump(done, open(out, 'w'))
        r = done[sid]; b = r['base']
        print(f"{ds} {learner} {sid:14s} base {b:.4f}  INV|T {r['INV|T']/b-1:+.3f}  INV|O {r['INV|O']/b-1:+.3f}  HP|O {r['HP|O']/b-1:+.3f}  HP|O* {r['HP|O*']/b-1:+.3f} ({time.time()-t:.0f}s)", flush=True)
    print('DONE', ds, learner)
