"""Re-run the equivalence oracle on every PAT miss and record which intervention first
changed an audited output, and whether any verdict came from a pipeline exception.

usage: python oracle_detail.py <results_dir>
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, compare, _REAL_READ_CSV
from data_temporal import load_panels, series_frame, audit_rows, IMPUTING_CONFIGS
from corpus_temporal import PANELS

R = sys.argv[1]
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
P = load_panels(os.path.join(BASE, 'data', 'panels.npy'))


def temporal_detail(pipe, df, base, rows):
    rng = np.random.default_rng(12345)
    y0 = df['y'].to_numpy(dtype=float)
    for a in rows:
        keep = {'X_train': (lambda ids, a=a: ids <= a), 'X_test': (lambda ids, a=a: ids <= a),
                'y_train': (lambda ids, a=a: ids < a)}
        obs = ~np.isnan(y0); fut = np.arange(a, len(y0)); fut = fut[obs[fut]]
        past = y0[:a][obs[:a]]; big = 10 * np.nanmax(np.abs(past)) + 10
        cands = []
        for kind in ('zero', 'big', 'reverse', 'pastcopy'):
            y = y0.copy()
            if kind == 'zero': y[fut] = 0.0
            elif kind == 'big': y[fut] = big
            elif kind == 'reverse': y[fut] = y0[fut][::-1] + (1.0 if len(fut) == 1 else 0.0)
            else: y[fut] = past[-len(fut):] if len(fut) <= len(past) else past.mean()
            cands.append((kind, y))
        sd = np.std(np.diff(past)) * 5 + 1e-6
        for _ in range(50):
            y = y0.copy(); y[fut] = np.maximum(y0[fut] + rng.normal(0, sd, len(fut)), 0)
            cands.append(('gauss5', y))
        for kind, y in cands:
            d = df.copy(); d['y'] = y
            try:
                pert = pipe.run(d)
            except Exception:
                return f'{kind}:exception'
            s, ch = compare(base, pert, keep)
            if s or any(v > 1e-9 for v in ch.values()):
                return kind
    return 'none'


out = collections.Counter()
recs = []
for l in open(os.path.join(R, 'temporal.jsonl')):
    r = json.loads(l)
    if r['op'] and not r['pat']['flagged'] and r.get('oracle_dependent'):
        df = series_frame(r['panel'], r['series'], P[r['panel']][r['series']], r['config'] in IMPUTING_CONFIGS)
        pipe = ScriptPipeline(os.path.join(BASE, 'sources', 'temporal', r['panel'], r['source'] + '.py'))
        k = temporal_detail(pipe, df, pipe.run(df), audit_rows(len(df), PANELS[r['panel']]['N_TEST']))
        out[(r['op'], k)] += 1
        recs.append(dict(panel=r['panel'], series=r['series'], source=r['source'], killed_by=k))
# tabular: reuse run_tabular.oracle but instrument
import run_tabular
meta = json.load(open(os.path.join(BASE, 'data', 'tabular_meta.json')))
for l in open(os.path.join(R, 'tabular.jsonl')):
    r = json.loads(l)
    if r['op'] and not r['pat']['flagged'] and r.get('oracle_dependent'):
        df = _REAL_READ_CSV(os.path.join(BASE, 'data', os.path.basename(meta[r['dataset']]['path']))).sample(frac=1.0, random_state=r['split']).reset_index(drop=True)
        pipe = ScriptPipeline(os.path.join(BASE, 'sources', 'tabular', r['dataset'], r['source'] + '.py'))
        base = pipe.run(df)
        orig = pipe.run
        exc = {'n': 0}
        def wrapped(d, orig=orig):
            try:
                return orig(d)
            except Exception:
                exc['n'] += 1
                raise
        pipe.run = wrapped
        dep = run_tabular.oracle(pipe, df, base)
        k = 'exception' if exc['n'] else ('structured-or-random' if dep else 'none')
        out[(r['op'] + '(tab)', k)] += 1
        recs.append(dict(dataset=r['dataset'], split=r['split'], source=r['source'], killed_by=k))
json.dump(dict(summary={f'{a}|{b}': v for (a, b), v in out.items()}, records=recs),
          open(os.path.join(R, 'oracle_detail.json'), 'w'), indent=1)
print(dict(out))
