"""PAT with a stochastic generator. The inventory simulator draws lead times at random and
is run in three ways: 'seeded' (fixed seed, draws consumed when orders are placed),
'unseeded' (fresh randomness on every run), and 'crn' (common random numbers: one lead time
pre-drawn per period from a fixed seed). B=5 draws at scale 1, two origins per series.
Output: results/pat_crn.json"""
import json, numpy as np, gens
from pat import pat
from run_bench import CFG
panels = np.load('panels.npy', allow_pickle=True).item(); out = {}
for mode in ('seeded', 'unseeded', 'crn'):
    gens.INV_MODE = mode; rng = np.random.default_rng(31)
    for ds, c in CFG.items():
        for sid, y in panels[ds].items():
            T = len(y); origins = [T-1-c['H']*(c['R']-r) for r in range(c['R'])]; t0 = origins[0]+1
            for o in (origins[0], origins[-1]):
                for p in ('T', 'O'):
                    out[f'{mode}|{ds}|{sid}|{o}|{p}'] = pat(y, c, o, t0, 'INV', p, 5, rng)
gens.INV_MODE = 'seeded'
json.dump(out, open('results/pat_crn.json', 'w'))
for mode in ('seeded', 'unseeded', 'crn'):
    for p in ('T', 'O'):
        H = np.array([v for k, v in out.items() if k.startswith(mode+'|') and k.endswith('|'+p)])
        print(mode, p, 'flag rate (any of 5)', H.any(1).mean().round(4))
