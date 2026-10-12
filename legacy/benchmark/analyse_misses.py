"""Investigate the eight audits that PAT (B=5, scale 1) missed in the controlled-generator
benchmark. All eight concern the target-indexed newsvendor rule (NV|T) on M4-Daily.

For each missed audit (series, origin o), the script reports:
  - the margin: for every target row o+k, how far the largest future value in the 28-value
    window lies below the order statistics that define the 0.8 quantile, in units of the
    perturbation standard deviation;
  - detection by the supplementary runs at B=20 and scale 1 or 3 (results/pat_nv.json);
  - detection with B=5 at scales 1, 2 and 3, rerun here with a fixed seed.

usage: run from the legacy benchmark folder:  python analyse_misses.py <out.json>
"""
import json, sys
import numpy as np
from run_bench import CFG
from pat import pat

P = json.load(open('results/pat.json'))
NV = json.load(open('results/pat_nv.json'))
panels = np.load('panels.npy', allow_pickle=True).item()
misses = [k for k, v in P.items() if not v['admissible'] and not any(v['hits'])]
out = {}
for key in misses:
    ds, sid, o, g, proto = key.split('|')
    o = int(o); c = CFG[ds]; y_raw = panels[ds][sid]
    T = len(y_raw); origins = [T - 1 - c['H'] * (c['R'] - r) for r in range(c['R'])]
    t0 = origins[0] + 1
    sd = np.std(np.diff(y_raw[:t0]))
    margins = []
    for k in range(1, c['H'] + 1):
        w = y_raw[o + k - 27: o + k + 1]                 # 28-value window ending at o+k
        fut = y_raw[o + 1: o + k + 1]
        srt = np.sort(w)
        lo_os = srt[21]                                  # 22nd order statistic (0-based 21)
        margins.append(float((lo_os - fut.max()) / sd))
    rec = dict(series=sid, origin=o, n_future_rows=c['H'],
               min_margin_sd=float(min(margins)), median_margin_sd=float(np.median(margins)),
               share_rows_all_future_below=float(np.mean(np.array(margins) > 0)),
               b20_scale1=bool(any(NV.get(f'1.0|{ds}|{sid}|{o}', []))),
               b20_scale3=bool(any(NV.get(f'3.0|{ds}|{sid}|{o}', []))))
    for s in (1.0, 2.0, 3.0):
        rng = np.random.default_rng(99)
        rec[f'b5_scale{int(s)}'] = bool(any(pat(y_raw, c, o, t0, 'NV', 'T', 5, rng, s)))
    out[key] = rec
    print(key, rec, flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
