"""Run PAT on the library-transformer cases (RQ4) and compare with documented semantics.

usage: python run_library.py <sources_root> <panels.npy> <tabular_meta.json> <out.jsonl>
"""
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pat_core import ScriptPipeline, pat_temporal, pat_tabular, _REAL_READ_CSV   # noqa: E402

N_TEST = 56
SERIES = [('M5', 0), ('M4D', 0), ('M4H', 0), ('ELEC', 0)]   # first series of each panel
TAB_SPLITS = 5


def masked_frame(y, key):
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    audits = [n - N_TEST, n - 1]
    cand = np.setdiff1d(np.arange(61, n), audits)
    y[rng.choice(cand, size=int(0.02 * n), replace=False)] = np.nan
    for a in audits:
        y[a - 1] = np.nan
    return pd.DataFrame({'y': y})


def main():
    root, panels_path, meta_path, out = sys.argv[1:5]
    idx = json.load(open(os.path.join(root, 'library_index.json')))
    P = np.load(panels_path, allow_pickle=True).item()
    meta = json.load(open(meta_path))
    with open(out, 'w') as f:
        for r in idx:
            pipe = ScriptPipeline(r['path'] if os.path.isabs(r['path']) else os.path.join(root, r['path']))
            if r['kind'] == 'temporal':
                for panel, i in SERIES:
                    sid = list(P[panel])[i]
                    y = P[panel][sid]
                    df = masked_frame(y, f'{panel}|{sid}') if r['masked'] else \
                        pd.DataFrame({'y': np.asarray(y, float)})
                    rec = dict(r, panel=panel, series=str(sid))
                    rec.pop('path')
                    try:
                        res = pat_temporal(pipe, df, [len(df) - N_TEST, len(df) - 1], B=5,
                                           scale=2.0, seed=zlib.crc32(f"{r['case']}|{sid}".encode()))
                        rec.update(flagged=res['flagged'], seconds=res['seconds'],
                                   hits=[a['hits'] for a in res['audits']])
                    except Exception as e:
                        rec['error'] = repr(e)[:300]
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
            else:
                gp = meta['german']['path']
                raw = _REAL_READ_CSV(gp if os.path.isabs(gp) else os.path.join(os.path.dirname(meta_path), gp))
                for s in range(TAB_SPLITS):
                    df = raw.sample(frac=1.0, random_state=s).reset_index(drop=True)
                    rec = dict(r, dataset='german', split=s)
                    rec.pop('path')
                    try:
                        res = pat_tabular(pipe, df, B=5,
                                          seed=zlib.crc32(f"{r['case']}|{r['pattern']}|{s}".encode()))
                        rec.update(flagged=res['flagged'], seconds=res['seconds'], hits=res['hits'])
                    except Exception as e:
                        rec['error'] = repr(e)[:300]
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
            print(r['case'], r.get('pattern', ''), 'done', flush=True)


if __name__ == '__main__':
    main()
