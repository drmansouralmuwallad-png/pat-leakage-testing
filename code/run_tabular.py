"""Run PAT and the equivalent-mutant oracle on every tabular instance
(dataset x split x pipeline source). Writes one JSON line per instance.

usage: python run_tabular.py <sources_root> <tabular_meta.json> <out.jsonl> [n_splits] [dump_dir]
"""
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, pat_tabular, compare, _REAL_READ_CSV   # noqa: E402

MAIN = dict(B=5, tol=1e-9)


def oracle(pipe, df, base, label='label', seed=12345):
    """Equivalent-mutant screening with structured interventions on the unavailable
    information: adversarial test labels and extreme feature values for the rows in B."""
    rng = np.random.default_rng(seed)
    test_ids = np.unique(base['y_test'][1])
    perm = np.random.default_rng(0).permutation(test_ids)
    a_ids, b_ids = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    keep = {'X_train': (lambda ids: np.ones(len(ids), bool)),
            'y_train': (lambda ids: np.ones(len(ids), bool)),
            'X_test': (lambda ids: np.isin(ids, a_ids))}
    feat_cols = [c for c in df.columns if c != label]
    num_cols = [c for c in feat_cols if df[c].dtype.kind in 'fi']
    y0 = df.loc[test_ids, label].to_numpy()
    label_variants = [1 - y0, np.ones_like(y0), np.zeros_like(y0)]
    for c in num_cols[:15]:
        v = df.loc[test_ids, c].to_numpy(dtype=float)
        label_variants.append((v > np.nanmedian(v)).astype(int))
        label_variants.append((v <= np.nanmedian(v)).astype(int))
    feature_variants = ['keep', 'extreme']
    cands = [(lv, fv) for lv in label_variants for fv in feature_variants]
    for _ in range(30):
        cands.append((rng.integers(0, 2, len(y0)), 'resample'))
    for lv, fv in cands:
        d = df.copy()
        d.loc[test_ids, label] = lv
        if fv == 'extreme' and num_cols:
            d.loc[b_ids, num_cols] = d.loc[b_ids, num_cols].to_numpy(dtype=float) * 10 + 100
        elif fv == 'resample':
            donors = rng.integers(0, len(d), size=len(b_ids))
            d.loc[b_ids, feat_cols] = d.iloc[donors][feat_cols].to_numpy()
        try:
            pert = pipe.run(d)
        except Exception:
            return True
        struct, ch = compare(base, pert, keep)
        if struct or any(v > 1e-9 for v in ch.values()):
            return True
    return False


def main():
    root, meta_path, out_path = sys.argv[1:4]
    n_splits = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    dump = sys.argv[5] if len(sys.argv) > 5 else None
    if dump:
        os.makedirs(dump, exist_ok=True)
    idx = json.load(open(os.path.join(root, 'tabular_index.json')))
    meta = json.load(open(meta_path))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            done.add((d['dataset'], d['split'], d['source']))
    t0 = time.time()
    with open(out_path, 'a') as f:
        for ds, m in meta.items():
            raw = _REAL_READ_CSV(m['path'] if os.path.isabs(m['path']) else os.path.join(os.path.dirname(meta_path), m['path']))
            for s in range(n_splits):
                df = raw.sample(frac=1.0, random_state=s).reset_index(drop=True)
                for r in [r for r in idx if r['dataset'] == ds]:
                    if (ds, s, r['source']) in done:
                        continue
                    pipe = ScriptPipeline(r['path'] if os.path.isabs(r['path']) else os.path.join(root, r['path']))
                    rec = dict(dataset=ds, split=s, source=r['source'], config=r['config'],
                               op=r['op'], admissible=r['op'] is None)
                    tb = time.perf_counter()
                    base = pipe.run(df)
                    rec['t_base'] = time.perf_counter() - tb
                    res = pat_tabular(pipe, df, seed=zlib.crc32(f'{ds}|{s}|{r["source"]}'.encode()),
                                      base=base, **MAIN)
                    rec['pat'] = dict(flagged=res['flagged'], hits=res['hits'],
                                      seconds=res['seconds'], n_runs=res['n_runs'],
                                      structural=res['structural'],
                                      mrc=min(res['max_rel_change'], 1e300),
                                      n_flagged_cols=len(res['flagged_cols']))
                    if r['op'] is not None and not res['flagged']:
                        rec['oracle_dependent'] = oracle(pipe, df, base)
                    if dump:
                        np.savez_compressed(
                            os.path.join(dump, f"{ds}__{s}__{r['source']}.npz"),
                            Xtr=base['X_train'][0], ytr=base['y_train'][0].ravel(),
                            Xte=base['X_test'][0], yte=base['y_test'][0].ravel(),
                            cols=np.array(base['X_train'][2]))
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                print(f'{ds} split {s} done, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
