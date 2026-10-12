"""Supplementary PAT run on the piecewise-linear re-solved newsvendor generator under
target indexing: detection as a function of the number of draws B and of the
perturbation scale (multiples of the standard deviation of first differences)."""
import json, numpy as np
from pat import pat
from run_bench import CFG
panels = np.load('panels.npy', allow_pickle=True).item(); out = {}
for scale in (1.0, 3.0):
    rng = np.random.default_rng(13)
    for ds, c in CFG.items():
        for sid, y in panels[ds].items():
            T = len(y); origins = [T-1-c['H']*(c['R']-r) for r in range(c['R'])]; t0 = origins[0]+1
            for o in (origins[0], origins[-1]):
                out[f'{scale}|{ds}|{sid}|{o}'] = pat(y, c, o, t0, 'NV', 'T', 20, rng, scale)
json.dump(out, open('results/pat_nv.json', 'w'))
for scale in ('1.0', '3.0'):
    H = np.array([v for k, v in out.items() if k.startswith(scale)])
    print(scale, 'single-draw', H.mean().round(4), 'any of 5', H[:, :5].any(1).mean().round(4),
          'any of 20', H.any(1).mean().round(4), 'by ds', {ds: np.array([v for k, v in out.items() if k.startswith(scale+'|'+ds+'|')]).any(1).mean().round(3) for ds in CFG})
