"""Deepchecks baseline (run in a separate Python 3.11 environment with deepchecks 0.19.1).

Deepchecks' leakage-related checks are run on what the learner receives, with their default
conditions:
  FeatureLabelCorrelation        PPS of each feature with the label < 0.8 (training data)
  FeatureLabelCorrelationChange  PPS difference between training and test < 0.2, and
                                 training PPS < 0.7
  TrainTestSamplesMix            share of test samples that also appear in training data,
                                 with the default condition
An instance is flagged if any condition fails.

usage: python run_deepchecks.py <dump_dir> <out.jsonl> <task: regression|binary> [list_file]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from deepchecks.tabular import Dataset  # noqa: E402
from deepchecks.tabular.checks import (FeatureLabelCorrelation,  # noqa: E402
                                       FeatureLabelCorrelationChange, TrainTestSamplesMix)


def frames(z):
    cols = [str(c) for c in z['cols']]
    if len(cols) != z['Xtr'].shape[1]:
        cols = [f'c{j}' for j in range(z['Xtr'].shape[1])]
    tr = pd.DataFrame(z['Xtr'], columns=cols)
    te = pd.DataFrame(z['Xte'], columns=cols)
    tr['__label'] = z['ytr']
    te['__label'] = z['yte']
    return tr, te


def run_one(path, task):
    z = np.load(path, allow_pickle=True)
    tr, te = frames(z)
    kw = dict(label='__label', cat_features=[], label_type=task)
    dtr, dte = Dataset(tr, **kw), Dataset(te, **kw)
    t0 = time.perf_counter()
    out = {}
    r = FeatureLabelCorrelation(random_state=0).add_condition_feature_pps_less_than().run(dtr)
    out['flc'] = [c.is_pass() for c in r.conditions_results]
    out['flc_fail_features'] = sorted(k for k, v in r.value.items() if v >= 0.8)
    r = (FeatureLabelCorrelationChange(random_state=0)
         .add_condition_feature_pps_difference_less_than()
         .add_condition_feature_pps_in_train_less_than().run(dtr, dte))
    out['flcc'] = [c.is_pass() for c in r.conditions_results]
    r = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal().run(dtr, dte)
    out['mix'] = [c.is_pass() for c in r.conditions_results]
    out['flagged'] = not all(out['flc'] + out['flcc'] + out['mix'])
    out['seconds'] = time.perf_counter() - t0
    return out


if __name__ == '__main__':
    dump, out_path, task = sys.argv[1:4]
    files = sorted(os.listdir(dump))
    if len(sys.argv) > 4:
        keep = set(open(sys.argv[4]).read().split('\n'))
        files = [f for f in files if f in keep]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)['file'] for l in open(out_path)}
    t0 = time.time()
    with open(out_path, 'a') as f:
        for i, fn in enumerate(files):
            if fn in done:
                continue
            try:
                rec = run_one(os.path.join(dump, fn), task)
            except Exception as e:  # noqa
                rec = {'error': repr(e)[:300], 'flagged': None}
            rec['file'] = fn
            f.write(json.dumps(rec) + '\n')
            f.flush()
            if i % 50 == 0:
                print(i, len(files), f'{time.time() - t0:.0f}s', flush=True)
