"""Aggregate benchmark results into the tables and numbers used in the manuscript.
Writes results/summary.json and prints LaTeX-ready rows."""
import json, os, numpy as np, collections
from scipy.stats import wilcoxon
from gens import CELLS
DS = ['M5', 'M4D', 'M4H', 'ELEC']; LEARN = ['ridge', 'lgbm', 'xgb', 'mlp']
rng = np.random.default_rng(11)
def geo(x): return float(np.exp(np.mean(np.log(x))) - 1)
def boot_ci(x, B=10000):
    lx = np.log(x); idx = rng.integers(0, len(lx), (B, len(lx)))
    m = np.exp(lx[idx].mean(1)) - 1; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def wp(x, alt='two-sided'):
    d = np.log(x)
    if np.allclose(d, 0): return 1.0
    return float(wilcoxon(d, alternative=alt, zero_method='zsplit').pvalue)
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps)-r)*ps[i]); adj[i] = min(1.0, run)
    return adj
S = {}
for ds in DS:
    for L in LEARN:
        f = f'results/{ds}_{L}.json'
        if not os.path.exists(f): continue
        R = json.load(open(f)); ids = sorted(R)
        base = np.array([R[s]['base'] for s in ids]); sn = np.array([R[s]['snaive'] for s in ids])
        cells = {}
        for g, p, adm in CELLS:
            v = f'{g}|{p}'
            x = np.array([R[s][v] for s in ids]) / base
            cells[v] = dict(adm=adm, eff=geo(x), ci=boot_ci(x), wins=int(np.sum(x < 1)), n=len(x), p=wp(x))
            if v+'|s1' in R[ids[0]]:
                sh = np.array([R[s][v+'|s1'] for s in ids]) / np.array([R[s][v] for s in ids])
                cells[v].update(shift_med=float(np.median(sh)-1), shift_p=wp(sh, 'greater'))
        adj = holm([cells[v]['p'] for v in cells])
        for v, a in zip(cells, adj): cells[v]['p_holm'] = float(a)
        S[f'{ds}|{L}'] = dict(n=len(ids), base_vs_snaive=geo(base/sn), cells=cells)
json.dump(S, open('results/summary.json', 'w'), indent=1)

# ---------- printing
for key, blk in S.items():
    print(f"\n== {key}  n={blk['n']}  base vs seasonal naive {100*blk['base_vs_snaive']:+.1f}%")
    for v, c in blk['cells'].items():
        print(f"{v:9s} adm={int(c['adm'])} eff {100*c['eff']:+7.1f}% [{100*c['ci'][0]:+6.1f},{100*c['ci'][1]:+6.1f}] "
              f"wins {c['wins']:2d}/{c['n']} p {c['p']:.3g} holm {c['p_holm']:.3g}" + (f" | shift {100*c['shift_med']:+6.1f}% p {c['shift_p']:.3g}" if 'shift_med' in c else ''))
