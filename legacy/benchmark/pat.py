"""Perturbation admissibility test (PAT). For a forecast origin o the evaluation pipeline
is run twice: on the series y and on a copy whose values after o are perturbed. If any
input supplied to the model at prediction time changes, that input depended on data
unavailable at o and the evaluation is flagged. The pipeline is treated as a black box:
generators (including their calibration) are recomputed from scratch on the perturbed
copy. Results: results/pat.json."""
import json, numpy as np
from gens import CELLS, gen_matrix, gen_index
from run_bench import CFG, base_X

def inputs_at_origin(y_raw, c, o, t0, gname, proto):
    y = y_raw / y_raw[:t0].mean()
    Gm = gen_matrix(gname, proto, y, t0, c['m'])
    Jo = np.array([o])
    return np.vstack([np.hstack([base_X(y, Jo, k, c), Gm[gen_index(proto, Jo, k)]])
                      for k in range(1, c['H']+1)])

def pat(y_raw, c, o, t0, gname, proto, B, rng, scale=1.0):
    X0 = inputs_at_origin(y_raw, c, o, t0, gname, proto)
    sig = np.std(np.diff(y_raw[:t0])) * scale
    hits = []
    for _ in range(B):
        yp = y_raw.copy()
        yp[o+1:] = np.maximum(yp[o+1:] + sig*rng.normal(size=len(yp)-o-1), 0.0)
        X1 = inputs_at_origin(yp, c, o, t0, gname, proto)
        hits.append(bool(np.any(np.abs(X1-X0) > 1e-9*(1+np.abs(X0)))))
    return hits

if __name__ == '__main__':
    panels = np.load('panels.npy', allow_pickle=True).item()
    rng = np.random.default_rng(7); B = 5; out = {}
    for ds, c in CFG.items():
        for sid, y in panels[ds].items():
            T = len(y); origins = [T-1-c['H']*(c['R']-r) for r in range(c['R'])]
            t0 = origins[0]+1
            for o in (origins[0], origins[-1]):
                for gname, proto, adm in CELLS:
                    key = f'{ds}|{sid}|{o}|{gname}|{proto}'
                    out[key] = dict(admissible=adm, hits=pat(y, c, o, t0, gname, proto, B, rng))
        print('PAT done', ds, flush=True)
    json.dump(out, open('results/pat.json', 'w'))
    print('DONE')
