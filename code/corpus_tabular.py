"""Tabular pipeline corpus: seven clean classification pipelines and 11 single-defect mutants.
They are rendered as notebook-style scripts in the idioms of scikit-learn, pandas and
category_encoders.

Defect classes follow Kapoor and Narayanan (2023):
  L1.2  pre-processing fitted on training and test data together
  L1.3  feature selection fitted on training and test data together
  L1.4  duplicates across the training and test data (oversampling before the split)
  L2    illegitimate features, here target statistics computed with test labels
"""
import os

HEADER = '''import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample

df = pd.read_csv('data.csv')
NUM = {NUM}
CAT = {CAT}
'''
FILLCAT = "df[CAT] = df[CAT].fillna('missing')\n"
XY = "X = df.drop(columns=['label'])\ny = df['label']\n"
SPLIT = ("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, "
         "random_state=0)\n")
IMPUTE_POST = ("imp = SimpleImputer(strategy='median').fit(X_train[NUM])\n"
               "X_train[NUM] = imp.transform(X_train[NUM])\n"
               "X_test[NUM] = imp.transform(X_test[NUM])\n")
SCALE_POST = ("sc = StandardScaler().fit(X_train[NUM])\n"
              "X_train[NUM] = sc.transform(X_train[NUM])\n"
              "X_test[NUM] = sc.transform(X_test[NUM])\n")
SCALE_POST_ARR = ("sc = StandardScaler().fit(X_train)\n"
                  "X_train = sc.transform(X_train)\n"
                  "X_test = sc.transform(X_test)\n")
OHE_POST = ("ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[CAT])\n"
            "X_train = np.hstack([X_train[NUM].to_numpy(), ohe.transform(X_train[CAT])])\n"
            "X_test = np.hstack([X_test[NUM].to_numpy(), ohe.transform(X_test[CAT])])\n")
NUM_ONLY_POST = "X_train = X_train[NUM].to_numpy()\nX_test = X_test[NUM].to_numpy()\n"
MODEL = ("clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)\n"
         "acc = clf.score(X_test, y_test)\n")

PRE = {
    'IMPUTE_ALL': "X[NUM] = SimpleImputer(strategy='median').fit_transform(X[NUM])\n",
    'SCALE_ALL': "X[NUM] = StandardScaler().fit_transform(X[NUM])\n",
    'SCALE_PANDAS_ALL': "X[NUM] = (X[NUM] - X[NUM].mean()) / X[NUM].std()\n",
    'TE_ALL': "X[CAT] = ce.TargetEncoder(cols=CAT).fit_transform(X[CAT], y)\n",
    'TE_LOO_ALL': "X[CAT] = ce.LeaveOneOutEncoder(cols=CAT).fit_transform(X[CAT], y)\n",
    'TE_PANDAS_ALL': ("for c in CAT:\n"
                      "    X[c] = X[c].map(df.groupby(c)['label'].mean())\n"),
    'KNNIMPUTE_ALL': "X[NUM] = KNNImputer(n_neighbors=5).fit_transform(X[NUM])\n",
}

CONFIGS = {
    'D1_impute_scale_ohe': ['IMPUTE_ALL', 'SCALE_ALL', 'SCALE_PANDAS_ALL'],
    'D2_target_encoding': ['TE_ALL', 'TE_LOO_ALL', 'TE_PANDAS_ALL'],
    'D3_selection': ['SEL_ALL'],
    'D4_pca': ['PCA_ALL_CONCAT'],
    'D5_knn_impute': ['KNNIMPUTE_ALL'],
    'D6_oversample': ['OVERSAMPLE_BEFORE_SPLIT'],
    'D7_sklearn_target_encoder': ['TE_SK_ALL'],
}
NEEDS_CAT = {'D2_target_encoding', 'D7_sklearn_target_encoder'}

OPERATORS = {
    'IMPUTE_ALL': dict(taxonomy='L1.2', mechanism='weak (global statistic)', label_free=True),
    'SCALE_ALL': dict(taxonomy='L1.2', mechanism='weak (global statistic)', label_free=True),
    'SCALE_PANDAS_ALL': dict(taxonomy='L1.2', mechanism='weak (global statistic)',
                             label_free=True),
    'KNNIMPUTE_ALL': dict(taxonomy='L1.2', mechanism='conditional', label_free=True),
    'PCA_ALL_CONCAT': dict(taxonomy='L1.2', mechanism='weak (global statistic)',
                           label_free=True),
    'SEL_ALL': dict(taxonomy='L1.3', mechanism='nonlinear (piecewise)', label_free=False),
    'TE_ALL': dict(taxonomy='L2', mechanism='direct', label_free=False),
    'TE_LOO_ALL': dict(taxonomy='L2', mechanism='direct', label_free=False),
    'TE_PANDAS_ALL': dict(taxonomy='L2', mechanism='direct', label_free=False),
    'TE_SK_ALL': dict(taxonomy='L2', mechanism='direct', label_free=False),
    'OVERSAMPLE_BEFORE_SPLIT': dict(taxonomy='L1.4', mechanism='direct', label_free=False),
}


def render(config, op, NUM, CAT):
    src = HEADER.format(NUM=repr(NUM), CAT=repr(CAT))
    if CAT:
        src += FILLCAT
    src += XY
    k = max(2, len(NUM) // 2)
    if config == 'D1_impute_scale_ohe':
        pre = PRE[op] if op else ''
        post = ('' if op == 'IMPUTE_ALL' else IMPUTE_POST)
        post += ('' if op in ('SCALE_ALL', 'SCALE_PANDAS_ALL') else SCALE_POST)
        post += OHE_POST if CAT else NUM_ONLY_POST
        if op in ('SCALE_ALL', 'SCALE_PANDAS_ALL'):
            # scaling before imputation is the usual order when it is done up front
            post = IMPUTE_POST + (OHE_POST if CAT else NUM_ONLY_POST)
        src += pre + SPLIT + post
    elif config == 'D2_target_encoding':
        pre = PRE[op] if op else ''
        post = '' if op else ("te = ce.TargetEncoder(cols=CAT).fit(X_train[CAT], y_train)\n"
                              "X_train[CAT] = te.transform(X_train[CAT])\n"
                              "X_test[CAT] = te.transform(X_test[CAT])\n")
        post += IMPUTE_POST + SCALE_POST
        post += ("X_train = X_train[NUM + CAT].to_numpy(dtype=float)\n"
                 "X_test = X_test[NUM + CAT].to_numpy(dtype=float)\n")
        src += pre + SPLIT + post
    elif config == 'D3_selection':
        src += "X = X[NUM].fillna(0.0)\n"
        if op == 'SEL_ALL':
            src += (f"sel = SelectKBest(f_classif, k={k}).fit(X, y)\n"
                    "X = X[X.columns[sel.get_support()]]\n")
            src += SPLIT + SCALE_POST_ARR
        else:
            src += SPLIT + (f"sel = SelectKBest(f_classif, k={k}).fit(X_train, y_train)\n"
                            "X_train = sel.transform(X_train)\n"
                            "X_test = sel.transform(X_test)\n") + SCALE_POST_ARR
    elif config == 'D4_pca':
        src += "X = X[NUM].fillna(0.0)\n" + SPLIT + SCALE_POST_ARR
        if op == 'PCA_ALL_CONCAT':
            src += "pca = PCA(n_components=3, random_state=0).fit(np.vstack([X_train, X_test]))\n"
        else:
            src += "pca = PCA(n_components=3, random_state=0).fit(X_train)\n"
        src += "X_train = pca.transform(X_train)\nX_test = pca.transform(X_test)\n"
    elif config == 'D5_knn_impute':
        src += "X = X[NUM]\n"
        if op == 'KNNIMPUTE_ALL':
            src += PRE[op] + SPLIT
        else:
            src += SPLIT + ("knn = KNNImputer(n_neighbors=5).fit(X_train)\n"
                            "X_train = pd.DataFrame(knn.transform(X_train), index=X_train.index,"
                            " columns=NUM)\n"
                            "X_test = pd.DataFrame(knn.transform(X_test), index=X_test.index,"
                            " columns=NUM)\n")
        src += SCALE_POST + NUM_ONLY_POST
    elif config == 'D6_oversample':
        src += "X = X[NUM].fillna(0.0)\n"
        up = ("minority = data[data['label'] == 1]\n"
              "majority = data[data['label'] == 0]\n"
              "up = resample(minority, replace=True, n_samples=len(majority), random_state=0)\n"
              "data = pd.concat([majority, up])\n")
        if op == 'OVERSAMPLE_BEFORE_SPLIT':
            src += ("data = pd.concat([X, y], axis=1)\n" + up +
                    "X, y = data.drop(columns=['label']), data['label']\n" + SPLIT)
        else:
            src += SPLIT + ("data = pd.concat([X_train, y_train], axis=1)\n" + up +
                            "X_train, y_train = data.drop(columns=['label']), data['label']\n")
        src += SCALE_POST + NUM_ONLY_POST
    elif config == 'D7_sklearn_target_encoder':
        if op == 'TE_SK_ALL':
            src += ("enc = TargetEncoder(random_state=0).fit_transform(X[CAT], y)\n"
                    "X = pd.concat([X[NUM], pd.DataFrame(enc, index=X.index, columns=CAT)],"
                    " axis=1)\n")
            src += SPLIT + IMPUTE_POST + SCALE_POST + (
                "X_train = X_train[NUM + CAT].to_numpy(dtype=float)\n"
                "X_test = X_test[NUM + CAT].to_numpy(dtype=float)\n")
        else:
            src += SPLIT + ("te = TargetEncoder(random_state=0)\n"
                            "tr_cat = te.fit_transform(X_train[CAT], y_train)\n"
                            "te_cat = te.transform(X_test[CAT])\n") + IMPUTE_POST + SCALE_POST + (
                "X_train = np.hstack([X_train[NUM].to_numpy(), tr_cat])\n"
                "X_test = np.hstack([X_test[NUM].to_numpy(), te_cat])\n")
    return src + MODEL


def write_sources(root, meta):
    rows = []
    for ds, m in meta.items():
        d = os.path.join(root, 'tabular', ds)
        os.makedirs(d, exist_ok=True)
        for cfg, ops in CONFIGS.items():
            if cfg in NEEDS_CAT and not m['cat']:
                continue
            for op in [None] + ops:
                sid = f"{cfg}__{op or 'clean'}"
                path = os.path.join(d, sid + '.py')
                with open(path, 'w') as f:
                    f.write(render(cfg, op, m['num'], m['cat']))
                rows.append(dict(dataset=ds, source=sid, config=cfg, op=op, path=path))
    return rows


if __name__ == '__main__':
    import json
    import sys
    meta = json.load(open(sys.argv[2]))
    rows = write_sources(sys.argv[1], meta)
    json.dump(rows, open(os.path.join(sys.argv[1], 'tabular_index.json'), 'w'), indent=1)
    print(len(rows), 'tabular sources written')
