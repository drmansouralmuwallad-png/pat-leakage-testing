"""Temporal pipeline corpus: six clean feature pipelines and 20 single-defect mutants.

Each pipeline is rendered as a plain, notebook-style Python script so that dynamic tests
(PAT, shift check, Deepchecks) and static tools (the time-offset lint, the leakage analyser of
Yang et al.) all analyse exactly the same code. A mutant differs from its clean parent only in
the code of one stage. The difference is a leakage defect from the taxonomy of Kapoor and
Narayanan (2023).

Every evaluation is one-step-ahead: the input row for target y[t] may use y[:t] only. The last
N_TEST rows form the test block and all fitted stages are fitted on the training block.
"""
import os

HEADER = '''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter

df = pd.read_csv('series.csv')
M = {M}
N_TEST = {N_TEST}
n_fit = len(df) - N_TEST
'''

# Stage code. Each stage has a clean variant and, for some mutation operators, a defective
# variant. Stages marked 'post' run after the train/test split.
IMP = {
    'clean': "y = df['y'].ffill()\n",
    'IMP_BFILL': "y = df['y'].bfill()\n",
    'IMP_INTERP': "y = df['y'].interpolate(method='linear', limit_direction='both')\n",
}
NOIMP = "y = df['y']\n"
WINS = {
    'clean': "lo, hi = y.iloc[:n_fit].quantile([0.01, 0.99])\ny_feat = y.clip(lo, hi)\n",
    'WINS_ALL': "lo, hi = y.quantile([0.01, 0.99])\ny_feat = y.clip(lo, hi)\n",
}
NOWINS = "y_feat = y\n"
LAGS = {
    'clean': ("feat = pd.DataFrame(index=df.index)\n"
              "for k in range(1, M + 1):\n"
              "    feat['lag_%d' % k] = y_feat.shift(k)\n"
              "feat['lag_2m'] = y_feat.shift(2 * M)\n"),
    'OFF_BY_ONE': ("feat = pd.DataFrame(index=df.index)\n"
                   "for k in range(0, M):\n"
                   "    feat['lag_%d' % k] = y_feat.shift(k)\n"
                   "feat['lag_2m'] = y_feat.shift(2 * M)\n"),
}
LAGS_MANY = ("feat = pd.DataFrame(index=df.index)\n"
             "for k in range(1, 2 * M + 1):\n"
             "    feat['lag_%d' % k] = y_feat.shift(k)\n")
ROLL = {
    'clean': ("feat['roll_mean'] = y_feat.shift(1).rolling(M).mean()\n"
              "feat['roll_std'] = y_feat.shift(1).rolling(M).std()\n"),
    'ROLL_CENTER': ("feat['roll_mean'] = y_feat.shift(1).rolling(M, center=True, min_periods=1).mean()\n"
                    "feat['roll_std'] = y_feat.shift(1).rolling(M, center=True, min_periods=2).std()\n"),
    'ROLL_NOSHIFT': ("feat['roll_mean'] = y_feat.rolling(M).mean()\n"
                     "feat['roll_std'] = y_feat.rolling(M).std()\n"),
}
SMOOTH = {
    'clean': "feat['smooth'] = y_feat.shift(1).ewm(alpha=0.3).mean()\n",
    'SMOOTH_SAVGOL': ("feat['smooth'] = pd.Series(savgol_filter(y_feat.to_numpy(), 2 * M + 1, 2),"
                      " index=df.index).shift(1)\n"),
}
DIFF = "feat['diff_m'] = y_feat.shift(1).diff(M)\n"
DECOMP = {
    'clean': ("dec = seasonal_decompose(y_feat, period=M, two_sided=False)\n"
              "feat['trend'] = dec.trend.shift(1)\n"),
    'DECOMP_TWOSIDED': ("dec = seasonal_decompose(y_feat, period=M, two_sided=True, "
                        "extrapolate_trend='freq')\n"
                        "feat['trend'] = dec.trend.shift(1)\n"),
}
FLAG = {
    'clean': ("thr = y_feat.iloc[:n_fit].quantile(0.9)\n"
              "feat['peak_1'] = (y_feat.shift(1) > thr).astype(float)\n"
              "feat['peak_m'] = (y_feat.shift(M) > thr).astype(float)\n"),
    'FLAG_ALL': ("thr = y_feat.quantile(0.9)\n"
                 "feat['peak_1'] = (y_feat.shift(1) > thr).astype(float)\n"
                 "feat['peak_m'] = (y_feat.shift(M) > thr).astype(float)\n"),
}
PHASE = "feat['phase'] = np.arange(len(df)) % M\n"
PROFILE = {
    'clean': {'pre': '', 'post': ("prof = y_train.groupby(X_train['phase']).mean()\n"
                                  "X_train['prof'] = X_train['phase'].map(prof)\n"
                                  "X_test['prof'] = X_test['phase'].map(prof)\n")},
    'PROFILE_ALL': {'pre': "feat['prof'] = feat['phase'].map(y.groupby(feat['phase']).mean())\n",
                    'post': ''},
}
WARMUP = "feat = feat.iloc[3 * M:]\ntarget = y.loc[feat.index]\n"
SPLIT = ("X_train, X_test, y_train, y_test = train_test_split(feat, target, "
         "test_size=N_TEST, shuffle=False)\n")
SCALE = {
    'clean': {'pre': '', 'post': (
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, "
        "columns=X_train.columns)\n"
        "X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, "
        "columns=X_test.columns)\n")},
    # pre-split form: the scaler is fitted on the whole feature table
    'SCALE_ALL': {'pre': ("feat = pd.DataFrame(StandardScaler().fit_transform(feat), "
                          "index=feat.index, columns=feat.columns)\n"), 'post': ''},
    # post-split form: the scaler is fitted on the training and test rows together
    'SCALE_ALL_CONCAT': {'pre': '', 'post': (
        "scaler = StandardScaler().fit(pd.concat([X_train, X_test]))\n"
        "X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, "
        "columns=X_train.columns)\n"
        "X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, "
        "columns=X_test.columns)\n")},
}
TSCALE = {
    'clean': {'pre': '', 'post': ("mu, sd = y_train.mean(), y_train.std()\n"
                                  "y_train = (y_train - mu) / sd\n")},
    'TSCALE_ALL': {'pre': "target = (target - target.mean()) / target.std()\n", 'post': ''},
}
SELECT = {
    'clean': {'pre': '', 'post': (
        "selector = SelectKBest(f_regression, k=8).fit(X_train, y_train)\n"
        "cols = X_train.columns[selector.get_support()]\n"
        "X_train, X_test = X_train[cols], X_test[cols]\n")},
    'SEL_ALL': {'pre': ("selector = SelectKBest(f_regression, k=8).fit(feat, target)\n"
                        "feat = feat[feat.columns[selector.get_support()]]\n"), 'post': ''},
}
PCA_ = {
    'clean': {'pre': '', 'post': (
        "pca = PCA(n_components=3, random_state=0).fit(X_train)\n"
        "X_train = pd.DataFrame(pca.transform(X_train), index=X_train.index)\n"
        "X_test = pd.DataFrame(pca.transform(X_test), index=X_test.index)\n")},
    'PCA_ALL_CONCAT': {'pre': '', 'post': (
        "pca = PCA(n_components=3, random_state=0).fit(pd.concat([X_train, X_test]))\n"
        "X_train = pd.DataFrame(pca.transform(X_train), index=X_train.index)\n"
        "X_test = pd.DataFrame(pca.transform(X_test), index=X_test.index)\n")},
}
MODEL = ("model = Ridge(alpha=1.0).fit(X_train, y_train)\n"
         "pred = model.predict(X_test)\n"
         "rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))\n")
MODEL_TSCALE = ("model = Ridge(alpha=1.0).fit(X_train, y_train)\n"
                "pred = model.predict(X_test) * sd + mu\n"
                "rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))\n")

# Clean configurations: ordered stages
CONFIGS = {
    'C1_basic':     ['imp', 'nowins', 'lags', 'roll', 'smooth', 'diff'],
    'C2_scaled':    ['noimp', 'nowins', 'lags', 'roll', '|', 'scale', 'tscale'],
    'C3_seasonal':  ['noimp', 'nowins', 'lags', 'phase', 'profile', 'decomp', 'flag'],
    'C4_selection': ['noimp', 'nowins', 'lags_many', 'roll', 'diff', 'select'],
    'C5_chain':     ['imp', 'wins', 'lags', 'roll', 'scale', 'pca'],
    'C6_robust':    ['noimp', 'wins', 'lags', 'roll', 'flag'],
}
# Mutation operators applicable to each configuration: (operator, stage it modifies)
MUTANTS = {
    'C1_basic': [('IMP_BFILL', 'imp'), ('IMP_INTERP', 'imp'), ('OFF_BY_ONE', 'lags'),
                 ('ROLL_NOSHIFT', 'roll'), ('ROLL_CENTER', 'roll'), ('SMOOTH_SAVGOL', 'smooth')],
    'C2_scaled': [('SCALE_ALL', 'scale'), ('TSCALE_ALL', 'tscale'), ('ROLL_CENTER', 'roll')],
    'C3_seasonal': [('PROFILE_ALL', 'profile'), ('DECOMP_TWOSIDED', 'decomp'),
                    ('FLAG_ALL', 'flag')],
    'C4_selection': [('SEL_ALL', 'select'), ('ROLL_NOSHIFT', 'roll')],
    'C5_chain': [('IMP_INTERP', 'imp'), ('WINS_ALL', 'wins'), ('SCALE_ALL_CONCAT', 'scale'),
                 ('PCA_ALL_CONCAT', 'pca')],
    'C6_robust': [('WINS_ALL', 'wins'), ('FLAG_ALL', 'flag')],
}

# Defect classes. 'taxonomy' follows Kapoor and Narayanan (2023): L1.2 pre-processing on
# training and test data, L1.3 feature selection on training and test data, L2 illegitimate
# feature, L3.1 temporal leakage. 'mechanism' is the property studied in RQ2.
OPERATORS = {
    'OFF_BY_ONE':       dict(taxonomy='L2/L3.1', mechanism='direct', cols=['lag_0']),
    'ROLL_NOSHIFT':     dict(taxonomy='L3.1', mechanism='direct', cols=['roll_mean', 'roll_std']),
    'ROLL_CENTER':      dict(taxonomy='L3.1', mechanism='hidden look-ahead',
                             cols=['roll_mean', 'roll_std']),
    'SMOOTH_SAVGOL':    dict(taxonomy='L3.1', mechanism='hidden look-ahead', cols=['smooth']),
    'DECOMP_TWOSIDED':  dict(taxonomy='L3.1', mechanism='hidden look-ahead', cols=['trend']),
    'IMP_BFILL':        dict(taxonomy='L3.1', mechanism='conditional', cols=None),
    'IMP_INTERP':       dict(taxonomy='L3.1', mechanism='conditional', cols=None),
    'SCALE_ALL':        dict(taxonomy='L1.2', mechanism='weak (global statistic)', cols='ALL'),
    'SCALE_ALL_CONCAT': dict(taxonomy='L1.2', mechanism='weak (global statistic)', cols='ALL'),
    'TSCALE_ALL':       dict(taxonomy='L1.2', mechanism='weak (global statistic)',
                             cols=['y_train:target']),
    'PROFILE_ALL':      dict(taxonomy='L2', mechanism='weak (global statistic)', cols=['prof']),
    'PCA_ALL_CONCAT':   dict(taxonomy='L1.2', mechanism='weak (global statistic)', cols='ALL'),
    'WINS_ALL':         dict(taxonomy='L1.2', mechanism='nonlinear (piecewise)', cols=None),
    'FLAG_ALL':         dict(taxonomy='L1.2', mechanism='nonlinear (piecewise)',
                             cols=['peak_1', 'peak_m']),
    'SEL_ALL':          dict(taxonomy='L1.3', mechanism='nonlinear (piecewise)', cols=None),
}
MULTI_STAGE_CONFIGS = {'C5_chain'}   # the leak passes through three or more later stages


def render(config, op=None, M=7, N_TEST=56):
    """Source code of a clean pipeline (op=None) or of a single-defect mutant."""
    stages = [s for s in CONFIGS[config] if s != '|']
    target_stage = dict(MUTANTS[config]).get(op) if op else None
    if op and target_stage is None:
        raise ValueError(f'{op} does not apply to {config}')

    def variant(stage):
        return op if stage == target_stage else 'clean'

    pre, post = [], []
    src = HEADER.format(M=M, N_TEST=N_TEST)
    for s in stages:
        v = variant(s)
        if s == 'imp':
            pre.append(IMP[v])
        elif s == 'noimp':
            pre.append(NOIMP)
        elif s == 'wins':
            pre.append(WINS[v])
        elif s == 'nowins':
            pre.append(NOWINS)
        elif s == 'lags':
            pre.append(LAGS[v])
        elif s == 'lags_many':
            pre.append(LAGS_MANY)
        elif s == 'roll':
            pre.append(ROLL[v])
        elif s == 'smooth':
            pre.append(SMOOTH[v])
        elif s == 'diff':
            pre.append(DIFF)
        elif s == 'decomp':
            pre.append(DECOMP[v])
        elif s == 'flag':
            pre.append(FLAG[v])
        elif s == 'phase':
            pre.append(PHASE)
    # stages that may act before the split (on the feature table) or after it
    feature_code = ''.join(pre) + WARMUP
    prepost = {'profile': PROFILE, 'scale': SCALE, 'tscale': TSCALE, 'select': SELECT,
               'pca': PCA_}
    pre_split, post_split = [], []
    for s in stages:
        if s in prepost:
            block = prepost[s][variant(s)]
            if s == 'profile' and block['pre']:
                # the profile feature is defined on the feature table before warm-up
                feature_code = ''.join(pre) + block['pre'] + WARMUP
            else:
                pre_split.append(block['pre'])
            post_split.append(block['post'])
    model = MODEL_TSCALE if ('tscale' in stages and variant('tscale') == 'clean') else MODEL
    src += feature_code + ''.join(pre_split) + SPLIT + ''.join(post_split) + model
    return src


def all_sources(M, N_TEST):
    """[(source_id, config, op, code)] for one panel."""
    out = []
    for cfg in CONFIGS:
        out.append((f'{cfg}__clean', cfg, None, render(cfg, None, M, N_TEST)))
        for op, _ in MUTANTS[cfg]:
            out.append((f'{cfg}__{op}', cfg, op, render(cfg, op, M, N_TEST)))
    return out


PANELS = {  # seasonal period and test-block length (the paper's horizon x origins)
    'M5':   dict(M=7,  N_TEST=56),
    'M4D':  dict(M=7,  N_TEST=84),
    'M4H':  dict(M=24, N_TEST=192),
    'ELEC': dict(M=24, N_TEST=192),
}


def write_sources(root):
    rows = []
    for panel, c in PANELS.items():
        d = os.path.join(root, 'temporal', panel)
        os.makedirs(d, exist_ok=True)
        for sid, cfg, op, code in all_sources(c['M'], c['N_TEST']):
            path = os.path.join(d, sid + '.py')
            with open(path, 'w') as f:
                f.write(code)
            rows.append(dict(panel=panel, source=sid, config=cfg, op=op, path=path))
    return rows


if __name__ == '__main__':
    import json
    import sys
    rows = write_sources(sys.argv[1])
    json.dump(rows, open(os.path.join(sys.argv[1], 'temporal_index.json'), 'w'), indent=1)
    print(len(rows), 'temporal sources written')
