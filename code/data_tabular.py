"""Tabular datasets for the non-temporal corpus.

adult          UCI Adult (Becker and Kohavi, 1996): a 5,000-row random sample of the 48,842
               rows. Label: income >50K.
german         UCI Statlog German Credit (Hofmann, 1994): 1,000 rows. Label: bad credit risk.
breast_cancer  UCI Breast Cancer Wisconsin, Diagnostic (Wolberg et al., 1995): 569 rows, as
               bundled with scikit-learn. Label: malignant.

Five per cent of the numeric values are set to missing completely at random with a fixed seed,
so that the imputation stages have work to do. Missing categorical values ('?' in Adult) are
kept as missing.
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'income']
GERMAN_NUM = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']


def build(raw_dir, out_dir, seed=2026):
    rng = np.random.default_rng(seed)
    meta = {}
    # Adult
    d = pd.read_csv(os.path.join(raw_dir, 'adult-all.csv'), header=None, names=ADULT_COLS,
                    na_values='?', skipinitialspace=True)
    d = d.sample(n=5000, random_state=seed).reset_index(drop=True)
    d['label'] = (d.pop('income').str.strip() == '>50K').astype(int)
    num = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat = [c for c in d.columns if c not in num + ['label']]
    meta['adult'] = (d, num, cat)
    # German credit
    g = pd.read_csv(os.path.join(raw_dir, 'german.csv'), header=None)
    g.columns = [f'A{i}' for i in range(1, 21)] + ['label']
    g['label'] = (g['label'] == 2).astype(int)
    cat = [c for c in g.columns if c not in GERMAN_NUM + ['label']]
    meta['german'] = (g, GERMAN_NUM, cat)
    # Breast cancer
    bc = load_breast_cancer(as_frame=True)
    b = bc.data.copy()
    b.columns = [c.replace(' ', '_') for c in b.columns]
    b['label'] = (bc.target == 0).astype(int)
    meta['breast_cancer'] = (b, [c for c in b.columns if c != 'label'], [])
    out = {}
    os.makedirs(out_dir, exist_ok=True)
    for name, (df, num, cat) in meta.items():
        df = df.copy()
        for c in num:
            df[c] = df[c].astype(float)
            m = rng.random(len(df)) < 0.05
            df.loc[m, c] = np.nan
        path = os.path.join(out_dir, f'tab_{name}.csv')
        df.to_csv(path, index=False)
        out[name] = dict(path=path, num=num, cat=cat, n=len(df),
                         base_rate=float(df['label'].mean()))
    return out


if __name__ == '__main__':
    import json
    import sys
    info = build(sys.argv[1], sys.argv[2])
    json.dump(info, open(os.path.join(sys.argv[2], 'tabular_meta.json'), 'w'), indent=1)
    for k, v in info.items():
        print(k, v['n'], 'rows', len(v['num']), 'numeric', len(v['cat']), 'categorical',
              'base rate', round(v['base_rate'], 3))
