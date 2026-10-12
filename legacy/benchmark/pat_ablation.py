"""PAT sensitivity and cost. At the last test origin of every series and for every cell,
the pipeline is rerun on 20 (inadmissible cells) or 3 (admissible cells) perturbed copies
at each perturbation scale (multiples of the s.d. of first differences). For every draw the
maximum relative input change max|X1-X0|/(1+|X0|) is stored, so detection can be evaluated
afterwards for any number of draws B and any tolerance eps. Wall-clock time of one pipeline
execution is recorded per cell. Output: results/pat_ablation.json"""
import json, time, numpy as np
from gens import CELLS
from run_bench import CFG
from pat import inputs_at_origin
SCALES = [0.25, 0.5, 1.0, 2.0, 3.0]
panels = np.load('panels.npy', allow_pickle=True).item(); rng = np.random.default_rng(23); out = {}
for ds, c in CFG.items():
    for sid, y in panels[ds].items():
        T = len(y); origins = [T-1-c['H']*(c['R']-r) for r in range(c['R'])]; t0 = origins[0]+1; o = origins[-1]
        sig = np.std(np.diff(y[:t0]))
        for g, p, adm in CELLS:
            t = time.perf_counter(); X0 = inputs_at_origin(y, c, o, t0, g, p); tx = time.perf_counter()-t
            rec = {}
            for s in SCALES:
                d = []
                for b in range(3 if adm else 20):
                    yp = y.copy(); yp[o+1:] = np.maximum(yp[o+1:] + s*sig*rng.normal(size=T-o-1), 0.0)
                    X1 = inputs_at_origin(yp, c, o, t0, g, p)
                    d.append(float(np.max(np.abs(X1-X0)/(1+np.abs(X0)))))
                rec[str(s)] = d
            out[f'{ds}|{sid}|{g}|{p}'] = dict(admissible=adm, sec_per_run=tx, maxrel=rec)
    print('ablation done', ds, flush=True)
json.dump(out, open('results/pat_ablation.json', 'w'))
