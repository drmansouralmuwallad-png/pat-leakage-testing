"""Run PAT, the shift-sensitivity check and the equivalent-mutant oracle on every temporal
instance (pipeline source x series). Writes one JSON line per instance.

usage: python run_temporal.py <sources_root> <panels.npy> <out.jsonl> [n_workers]
"""
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, pat_temporal, compare          # noqa: E402
from data_temporal import load_panels, series_frame, audit_rows, IMPUTING_CONFIGS  # noqa
from corpus_temporal import PANELS, OPERATORS                       # noqa: E402

MAIN = dict(B=5, scale=2.0, tol=1e-9)          # settings recommended in Section 4
PIECEWISE = {'WINS_ALL', 'FLAG_ALL', 'SEL_ALL'}
GRID_SCALES = [0.5, 1.0, 2.0, 3.0]
GRID_B = 10


def shift_check(base, n_val):
    """Shift-sensitivity heuristic computed inside the training block. A ridge model is fitted
    on the training rows before the last n_val rows and scored on those rows, once with the
    inputs as produced and once with every input row replaced by the previous row. The
    statistic is the relative increase in validation RMSE."""
    X, _, _ = base['X_train']
    y = base['y_train'][0].ravel()
    if len(y) <= n_val + 10:
        return None
    def score(Xm, ym):
        fit = Ridge(alpha=1.0).fit(Xm[:-n_val], ym[:-n_val])
        r = fit.predict(Xm[-n_val:]) - ym[-n_val:]
        return float(np.sqrt(np.mean(r ** 2)))
    Xc = np.nan_to_num(X)
    e0 = score(Xc[1:], y[1:])
    e1 = score(Xc[:-1], y[1:])
    return e1 / e0 - 1.0 if e0 > 0 else None


def oracle(pipe, df, base, rows):
    """Equivalent-mutant screening: stronger, structured interventions on the unavailable
    values. Returns True if any intervention changes an audited output."""
    rng = np.random.default_rng(12345)
    y0 = df['y'].to_numpy(dtype=float)
    for a in rows:
        keep = {'X_train': (lambda ids, a=a: ids <= a), 'X_test': (lambda ids, a=a: ids <= a),
                'y_train': (lambda ids, a=a: ids < a)}
        obs = ~np.isnan(y0)
        fut = np.arange(a, len(y0))
        fut = fut[obs[fut]]
        past = y0[:a][obs[:a]]
        big = 10 * np.nanmax(np.abs(past)) + 10
        candidates = []
        for kind in ('zero', 'big', 'reverse', 'pastcopy'):
            y = y0.copy()
            if kind == 'zero':
                y[fut] = 0.0
            elif kind == 'big':
                y[fut] = big
            elif kind == 'reverse':
                y[fut] = y0[fut][::-1] + (1.0 if len(fut) == 1 else 0.0)
            else:
                y[fut] = past[-len(fut):] if len(fut) <= len(past) else past.mean()
            candidates.append(y)
        sd = np.std(np.diff(past)) * 5 + 1e-6
        for _ in range(50):
            y = y0.copy()
            y[fut] = np.maximum(y0[fut] + rng.normal(0, sd, len(fut)), 0)
            candidates.append(y)
        for y in candidates:
            d = df.copy()
            d['y'] = y
            try:
                pert = pipe.run(d)
            except Exception:
                return True   # the pipeline's behaviour depends on the unavailable values
            struct, ch = compare(base, pert, keep)
            if struct or any(v > 1e-9 for v in ch.values()):
                return True
    return False


def run_series(job):
    panel, sid, rows = job
    P = load_panels(PANELS_PATH)
    y = P[panel][sid]
    c = PANELS[panel]
    out = []
    for r in rows:
        df = series_frame(panel, sid, y, r['config'] in IMPUTING_CONFIGS)
        pipe = ScriptPipeline(r['path'] if os.path.isabs(r['path']) else os.path.join(SOURCES_ROOT, r['path']))
        audits = audit_rows(len(df), c['N_TEST'])
        rec = dict(panel=panel, series=str(sid), source=r['source'], config=r['config'],
                   op=r['op'], admissible=r['op'] is None)
        t0 = time.perf_counter()
        try:
            base = pipe.run(df)
        except Exception as e:  # noqa
            rec['error'] = repr(e)
            out.append(rec)
            continue
        rec['t_base'] = time.perf_counter() - t0
        res = pat_temporal(pipe, df, audits, seed=zlib.crc32(f"{panel}|{sid}|{r['source']}".encode()), base=base,
                           **MAIN)
        rec['pat'] = dict(flagged=res['flagged'], seconds=res['seconds'], n_runs=res['n_runs'],
                          audits=[dict(row=a['row'], hits=a['hits'], flagged=a['flagged'],
                                       mrc=min(a['max_rel_change'], 1e300))
                                  for a in res['audits']])
        # one cheap draw at a small scale: does a single small perturbation suffice?
        r1 = pat_temporal(pipe, df, audits, B=1, scale=0.25, seed=7, base=base)
        rec['pat_b1_s025'] = r1['flagged']
        # shift-sensitivity heuristic
        t1 = time.perf_counter()
        rec['shift_stat'] = shift_check(base, c['N_TEST'])
        rec['t_shift'] = time.perf_counter() - t1 + rec['t_base']
        # sensitivity grid for piecewise defects
        if r['op'] in PIECEWISE:
            grid = {}
            for s in GRID_SCALES:
                g = pat_temporal(pipe, df, audits, B=GRID_B, scale=s,
                                 seed=1000 + int(s * 10), base=base)
                grid[str(s)] = [a['hits'] for a in g['audits']]
            rec['grid'] = grid
        # oracle for mutants that PAT did not flag
        if r['op'] is not None and not res['flagged']:
            rec['oracle_dependent'] = oracle(pipe, df, base, audits)
        # learner-facing outputs for Deepchecks
        if DUMP_DIR:
            np.savez_compressed(
                os.path.join(DUMP_DIR, f"{panel}__{sid}__{r['source']}.npz"),
                Xtr=base['X_train'][0], ytr=base['y_train'][0].ravel(),
                Xte=base['X_test'][0], yte=base['y_test'][0].ravel(),
                cols=np.array(base['X_train'][2]))
        out.append(rec)
    return out


PANELS_PATH = None
DUMP_DIR = None
SOURCES_ROOT = None


def init(p, d, s):
    global PANELS_PATH, DUMP_DIR, SOURCES_ROOT
    PANELS_PATH, DUMP_DIR, SOURCES_ROOT = p, d, s


if __name__ == '__main__':
    root, panels_path, out_path = sys.argv[1:4]
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    dump = sys.argv[5] if len(sys.argv) > 5 else None
    if dump:
        os.makedirs(dump, exist_ok=True)
    idx = json.load(open(os.path.join(root, 'temporal_index.json')))
    P = load_panels(panels_path)
    only = os.environ.get('ONLY_SERIES')     # optional "panel|sid,..." filter for testing
    jobs = []
    for panel in PANELS:
        rows = [r for r in idx if r['panel'] == panel]
        for sid in P[panel]:
            if only and f'{panel}|{sid}' not in only.split(','):
                continue
            jobs.append((panel, sid, rows))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            done.add((d['panel'], d['series']))
    jobs = [j for j in jobs if (j[0], str(j[1])) not in done]
    print(len(jobs), 'series to run', flush=True)
    t0 = time.time()
    with Pool(n_workers, initializer=init, initargs=(panels_path, dump, root)) as pool, \
            open(out_path, 'a') as f:
        for i, recs in enumerate(pool.imap_unordered(run_series, jobs)):
            for rec in recs:
                f.write(json.dumps(rec) + '\n')
            f.flush()
            print(f'{i + 1}/{len(jobs)} series done, {time.time() - t0:.0f}s', flush=True)
