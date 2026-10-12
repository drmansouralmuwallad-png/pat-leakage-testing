"""Perturbation Admissibility Test (PAT) for unmodified pipeline scripts.

A pipeline is an ordinary Python script that reads its data with pandas.read_csv and leaves
X_train, X_test, y_train (and y_test) in its global namespace, as a notebook would. PAT
executes the script as a black box. It replaces pandas.read_csv with a function that returns
an in-memory copy of the data, so the script itself is not modified.

The test reruns the script on copies of the data in which only the information unavailable at
prediction time has been changed. It then compares everything the learner receives: the
training inputs, the training labels and the prediction-time inputs. Any change means that
the pipeline depends on unavailable information.

Two information models are implemented:
  temporal   audit row a predicts y[a] from y[:a]. All observed values y[a:] are perturbed.
             The comparison covers every input row with id <= a and every training label with
             id < a.
  tabular    a test row is predicted from the training data and its own features. All test
             labels are resampled, and the features of a random half B of the test rows are
             replaced by other rows. The comparison covers the training inputs and labels and
             the inputs of the other half A of the test rows.
"""
import time
import warnings

import numpy as np
import pandas as pd

_REAL_READ_CSV = pd.read_csv


class ScriptPipeline:
    """Executes a pipeline script on a given DataFrame and returns what the learner sees."""

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.source = f.read()
        self.code = compile(self.source, path, 'exec')

    def run(self, data):
        def fake_read_csv(*args, **kwargs):
            return data.copy()
        pd.read_csv = fake_read_csv
        g = {'__name__': '__main__', '__file__': self.path}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exec(self.code, g)
        finally:
            pd.read_csv = _REAL_READ_CSV
        return normalise(g)


def _ids(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return np.asarray(obj.index)
    return None


def _arr(obj):
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=float).reshape(-1, 1)
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float)
    a = np.asarray(obj, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _cols(obj, p):
    if isinstance(obj, pd.DataFrame):
        return [str(c) for c in obj.columns]
    return [f'c{j}' for j in range(p)]


def normalise(g):
    """Learner-facing outputs as float arrays with row ids and column names."""
    out = {}
    y_tr_ids = _ids(g['y_train'])
    y_te_ids = _ids(g.get('y_test'))
    for name in ('X_train', 'X_test'):
        obj = g[name]
        a = _arr(obj)
        ids = _ids(obj)
        if ids is None:  # numpy output: rows are in the order of the labels
            ids = y_tr_ids if name == 'X_train' else y_te_ids
        out[name] = (a, np.asarray(ids), _cols(obj, a.shape[1]))
    out['y_train'] = (_arr(g['y_train']), np.asarray(y_tr_ids), ['target'])
    if 'y_test' in g:
        out['y_test'] = (_arr(g['y_test']), np.asarray(y_te_ids), ['target'])
    return out


def rel_change(a, b):
    """Elementwise relative change, treating two NaNs as equal and one NaN as a change."""
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    with np.errstate(invalid='ignore'):
        r = np.abs(a - b) / (1.0 + np.abs(a))
    r = np.where(both_nan, 0.0, r)
    r = np.where(one_nan, np.inf, r)
    return r


def compare(base, pert, keep):
    """Compare two output dicts on the rows selected by keep[name](ids).

    Returns (structural_change, {column_name: max_relative_change}).
    """
    changes = {}
    for name, sel in keep.items():
        a0, id0, c0 = base[name]
        a1, id1, c1 = pert[name]
        m0, m1 = sel(id0), sel(id1)
        if (c0 != c1 or m0.sum() != m1.sum()
                or not np.array_equal(id0[m0], id1[m1])):
            return True, {f'{name}:<structure>': np.inf}
        r = rel_change(a0[m0], a1[m1])
        if r.size:
            colmax = r.max(axis=0)
            for c, v in zip(c0, colmax):
                key = f'{name}:{c}'
                changes[key] = max(changes.get(key, 0.0), float(v))
    return False, changes


# ------------------------------------------------------------------ perturbation laws
def perturb_temporal(df, a, scale, rng, col='y', nonneg=True):
    """Gaussian noise on the observed values from row a onwards. The standard deviation is
    `scale` times that of the first differences of the observed values before a."""
    d = df.copy()
    y = d[col].to_numpy(dtype=float, copy=True)
    past = y[:a][~np.isnan(y[:a])]
    sd = np.std(np.diff(past)) * scale
    if sd == 0:
        sd = scale * (np.std(past) + 1.0)
    fut = np.arange(a, len(y))
    fut = fut[~np.isnan(y[fut])]
    y[fut] = y[fut] + rng.normal(0.0, sd, len(fut))
    if nonneg:
        y[fut] = np.maximum(y[fut], 0.0)
    d[col] = y
    return d


def perturb_tabular(df, test_ids, b_ids, rng, label='label'):
    """Resample every test label from the empirical training-label distribution, and replace
    the features of the rows in b_ids by those of randomly drawn rows of the data."""
    d = df.copy()
    train_mask = ~d.index.isin(test_ids)
    p_vals, p_cnt = np.unique(d.loc[train_mask, label].to_numpy(), return_counts=True)
    d.loc[test_ids, label] = rng.choice(p_vals, size=len(test_ids), p=p_cnt / p_cnt.sum())
    feat_cols = [c for c in d.columns if c != label]
    donors = rng.integers(0, len(d), size=len(b_ids))
    d.loc[b_ids, feat_cols] = d.iloc[donors][feat_cols].to_numpy()
    return d


# ------------------------------------------------------------------ the test
def pat_temporal(pipe, df, audit_rows, B=5, scale=1.0, tol=1e-9, seed=0, base=None,
                 stop_early=False):
    """PAT for a temporal pipeline. Returns a verdict with per-audit-row details."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if base is None:
        base = pipe.run(df)
    n_runs = 1
    audits = []
    for a in audit_rows:
        keep = {'X_train': (lambda ids, a=a: ids <= a),
                'X_test': (lambda ids, a=a: ids <= a),
                'y_train': (lambda ids, a=a: ids < a)}
        hits, colmax, struct_any = [], {}, False
        for b in range(B):
            pert = pipe.run(perturb_temporal(df, a, scale, rng))
            n_runs += 1
            struct, ch = compare(base, pert, keep)
            struct_any |= struct
            for k, v in ch.items():
                colmax[k] = max(colmax.get(k, 0.0), v)
            hit = struct or any(v > tol for v in ch.values())
            hits.append(bool(hit))
            if hit and stop_early:
                break
        flagged_cols = sorted(k for k, v in colmax.items() if v > tol)
        audits.append({'row': int(a), 'hits': hits, 'structural': struct_any,
                       'flagged': flagged_cols,
                       'max_rel_change': float(max(colmax.values())) if colmax else 0.0})
    return {'flagged': any(any(x['hits']) for x in audits), 'audits': audits,
            'n_runs': n_runs, 'seconds': time.perf_counter() - t0}


def pat_tabular(pipe, df, B=5, tol=1e-9, seed=0, base=None, label='label'):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if base is None:
        base = pipe.run(df)
    n_runs = 1
    test_ids = np.unique(base['y_test'][1])
    perm = rng.permutation(test_ids)
    a_ids, b_ids = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    keep = {'X_train': (lambda ids: np.ones(len(ids), bool)),
            'y_train': (lambda ids: np.ones(len(ids), bool)),
            'X_test': (lambda ids: np.isin(ids, a_ids))}
    hits, colmax, struct_any = [], {}, False
    for b in range(B):
        pert = pipe.run(perturb_tabular(df, test_ids, b_ids, rng, label))
        n_runs += 1
        struct, ch = compare(base, pert, keep)
        struct_any |= struct
        for k, v in ch.items():
            colmax[k] = max(colmax.get(k, 0.0), v)
        hits.append(bool(struct or any(v > tol for v in ch.values())))
    flagged_cols = sorted(k for k, v in colmax.items() if v > tol)
    return {'flagged': any(hits), 'hits': hits, 'structural': struct_any,
            'flagged_cols': flagged_cols,
            'max_rel_change': float(max(colmax.values())) if colmax else 0.0,
            'n_runs': n_runs, 'seconds': time.perf_counter() - t0}


def determinism_check(pipe, df, tol=1e-9):
    """Run the pipeline twice on identical data. Columns that differ reveal a nondeterministic
    stage (an unseeded random number generator), which must be fixed before PAT's verdict can
    be read as evidence of leakage."""
    a, b = pipe.run(df), pipe.run(df)
    keep = {k: (lambda ids: np.ones(len(ids), bool)) for k in ('X_train', 'X_test', 'y_train')}
    struct, ch = compare(a, b, keep)
    return sorted(k for k, v in ch.items() if v > tol) if not struct else ['<structure>']
