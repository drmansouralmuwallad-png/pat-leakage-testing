"""Library-transformer audit (RQ4).

Each case is a widely used library call in its common usage pattern, wrapped in a minimal
notebook-style pipeline. The expected verdict comes from the library's documented semantics,
which is independent of PAT.

Temporal cases: the transformer is applied to the whole series, its output is lagged by one
step (the analyst reads the value at t-1 to predict y[t]), and the pipeline is audited like the
temporal corpus. The output is admissible only if the transformer's value at t depends on y[:t+1]
alone.

Tabular cases: each transformer is used in two patterns. In pattern A it is fitted on all rows
before the split. In pattern B it is fitted on the training rows and applied to the test rows.
"""
import os

T_HEADER = '''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
{IMPORTS}
df = pd.read_csv('series.csv')
N_TEST = {N_TEST}
y = df['y']
'''
T_FOOTER = '''feat = pd.DataFrame({'f': np.asarray(f, dtype=float).ravel()}, index=df.index).shift(1)
feat = feat.iloc[60:].fillna(0.0)
target = y.loc[feat.index].ffill()
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test)
'''

# (case id, library, imports, code defining f, expected admissible?, masked series?, reason)
TEMPORAL = [
    ('pd_rolling_mean', 'pandas', '', 'f = y.rolling(7).mean()', True, False,
     'trailing window'),
    ('pd_rolling_center', 'pandas', '', 'f = y.rolling(7, center=True, min_periods=1).mean()',
     False, False, 'center=True labels the window at its centre'),
    ('pd_rolling_quantile', 'pandas', '', 'f = y.rolling(28).quantile(0.9)', True, False,
     'trailing window'),
    ('pd_ewm', 'pandas', '', 'f = y.ewm(alpha=0.3).mean()', True, False, 'recursive, causal'),
    ('pd_expanding', 'pandas', '', 'f = y.expanding().mean()', True, False, 'causal'),
    ('pd_diff', 'pandas', '', 'f = y.diff()', True, False, 'backward difference'),
    ('pd_pct_change', 'pandas', '', 'f = y.pct_change()', True, False, 'backward change'),
    ('pd_ffill', 'pandas', '', 'f = y.ffill()', True, True, 'propagates last valid value'),
    ('pd_bfill', 'pandas', '', 'f = y.bfill()', False, True, 'uses next valid value'),
    ('pd_interpolate', 'pandas', '', "f = y.interpolate(method='linear')", False, True,
     'interpolates between previous and next valid values'),
    ('pd_rank_pct', 'pandas', '', 'f = y.rank(pct=True)', False, False,
     'rank within the full series'),
    ('pd_qcut', 'pandas', '', 'f = pd.qcut(y, 5, labels=False, duplicates="drop")', False,
     False, 'bin edges from full-sample quantiles'),
    ('pd_zscore', 'pandas', '', 'f = (y - y.mean()) / y.std()', False, False,
     'full-sample mean and standard deviation'),
    ('np_gradient', 'numpy', '', 'f = np.gradient(y.to_numpy())', False, False,
     'central differences in the interior'),
    ('np_convolve_same', 'numpy', '', "f = np.convolve(y.to_numpy(), np.ones(7) / 7, mode='same')",
     False, False, "mode='same' centres the kernel"),
    ('sp_savgol', 'scipy', 'from scipy.signal import savgol_filter',
     'f = savgol_filter(y.to_numpy(), 15, 2)', False, False, 'centred polynomial window'),
    ('sp_filtfilt', 'scipy', 'from scipy.signal import butter, filtfilt',
     'b, a = butter(2, 0.1)\nf = filtfilt(b, a, y.to_numpy())', False, False,
     'forward-backward (zero-phase) filtering'),
    ('sp_lfilter', 'scipy', 'from scipy.signal import butter, lfilter',
     'b, a = butter(2, 0.1)\nf = lfilter(b, a, y.to_numpy())', True, False,
     'causal IIR filter'),
    ('sp_medfilt', 'scipy', 'from scipy.signal import medfilt',
     'f = medfilt(y.to_numpy(), 7)', False, False, 'centred median window'),
    ('sp_detrend', 'scipy', 'from scipy.signal import detrend', 'f = detrend(y.to_numpy())',
     False, False, 'least-squares line fitted to the whole series'),
    ('sp_uniform_filter1d', 'scipy', 'from scipy.ndimage import uniform_filter1d',
     'f = uniform_filter1d(y.to_numpy(), 7)', False, False, 'centred by default (origin=0)'),
    ('sp_gaussian_filter1d', 'scipy', 'from scipy.ndimage import gaussian_filter1d',
     'f = gaussian_filter1d(y.to_numpy(), 2)', False, False, 'symmetric kernel'),
    ('sm_decompose_2s_trend', 'statsmodels', 'from statsmodels.tsa.seasonal import seasonal_decompose',
     "f = seasonal_decompose(y, period=7, two_sided=True, extrapolate_trend='freq').trend",
     False, False, 'centred moving average'),
    ('sm_decompose_1s_trend', 'statsmodels', 'from statsmodels.tsa.seasonal import seasonal_decompose',
     'f = seasonal_decompose(y, period=7, two_sided=False).trend', True, False,
     'one-sided moving average'),
    ('sm_decompose_1s_seasonal', 'statsmodels',
     'from statsmodels.tsa.seasonal import seasonal_decompose',
     'f = seasonal_decompose(y, period=7, two_sided=False).seasonal', False, False,
     'seasonal component averages the detrended series over all periods'),
    ('sm_stl_trend', 'statsmodels', 'from statsmodels.tsa.seasonal import STL',
     'f = STL(y, period=7).fit().trend', False, False, 'LOESS smoothing, two-sided'),
    ('sm_hpfilter', 'statsmodels', 'from statsmodels.tsa.filters.hp_filter import hpfilter',
     'cycle, f = hpfilter(y, lamb=1600)', False, False, 'two-sided smoother'),
    ('sm_cffilter', 'statsmodels', 'from statsmodels.tsa.filters.cf_filter import cffilter',
     'f, trend = cffilter(y)', False, False, 'asymmetric full-sample band-pass filter'),
    ('sm_ets_fitted', 'statsmodels',
     'from statsmodels.tsa.holtwinters import ExponentialSmoothing',
     'f = ExponentialSmoothing(y.to_numpy()).fit().fittedvalues', False, False,
     'one-step fitted values, but the smoothing parameter is estimated on the whole series'),
    ('sm_ses_fixed', 'statsmodels', 'from statsmodels.tsa.holtwinters import SimpleExpSmoothing',
     "f = SimpleExpSmoothing(y.to_numpy(), initialization_method='known', initial_level=float(y.iloc[0]))"
     ".fit(smoothing_level=0.3, optimized=False).fittedvalues", True, False,
     'fixed parameters and a known initial level: causal recursion'),
    ('sm_uc_filtered', 'statsmodels', 'from statsmodels.tsa.statespace.structural import UnobservedComponents',
     "f = UnobservedComponents(y.to_numpy(), 'llevel').fit(disp=False).filtered_state[0]", False,
     False, 'filtered state, but the variances are estimated by maximum likelihood on the whole series'),
    ('sm_uc_smoothed', 'statsmodels', 'from statsmodels.tsa.statespace.structural import UnobservedComponents',
     "f = UnobservedComponents(y.to_numpy(), 'llevel').fit(disp=False).smoothed_state[0]", False,
     False, 'smoothed (two-sided) state'),
    ('sm_arima_fitted', 'statsmodels', 'from statsmodels.tsa.arima.model import ARIMA',
     'f = ARIMA(y.to_numpy(), order=(1, 0, 0)).fit().fittedvalues', False, False,
     'parameters estimated on the whole series'),
    ('sk_standard_scaler', 'scikit-learn', 'from sklearn.preprocessing import StandardScaler',
     'f = StandardScaler().fit_transform(y.to_frame())', False, False, 'fitted on the whole series'),
    ('sk_minmax_scaler', 'scikit-learn', 'from sklearn.preprocessing import MinMaxScaler',
     'f = MinMaxScaler().fit_transform(y.to_frame())', False, False, 'fitted on the whole series'),
    ('sk_kbins', 'scikit-learn', 'from sklearn.preprocessing import KBinsDiscretizer',
     "f = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf').fit_transform(y.to_frame())",
     False, False, 'bin edges fitted on the whole series'),
    ('sk_function_log1p', 'scikit-learn', 'from sklearn.preprocessing import FunctionTransformer',
     'f = FunctionTransformer(np.log1p).fit_transform(y.to_frame())', True, False,
     'stateless'),
    ('fe_lag_features', 'feature_engine', 'from feature_engine.timeseries.forecasting import LagFeatures',
     "f = LagFeatures(variables=['y'], periods=1).fit_transform(df[['y']])['y_lag_1']", True, False,
     'shifts by one period'),
    ('fe_window_features', 'feature_engine',
     'from feature_engine.timeseries.forecasting import WindowFeatures',
     "f = WindowFeatures(variables=['y'], window=7, functions=['mean']).fit_transform(df[['y']])['y_window_7_mean']",
     True, False, 'window shifted by one period by default'),
    ('fe_expanding_features', 'feature_engine',
     'from feature_engine.timeseries.forecasting import ExpandingWindowFeatures',
     "f = ExpandingWindowFeatures(variables=['y'], functions=['mean']).fit_transform(df[['y']])['y_expanding_mean']",
     True, False, 'expanding window shifted by one period by default'),
]

TAB_HEADER = '''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
{IMPORTS}
df = pd.read_csv('data.csv')
NUM = {NUM}
CAT = {CAT}
df[CAT] = df[CAT].fillna('missing')
X = df.drop(columns=['label'])
y = df['label']
'''
# (case id, library, imports, constructor, columns it acts on, uses labels?, label-free reason)
TABULAR = [
    ('sk_standard_scaler', 'scikit-learn', 'from sklearn.preprocessing import StandardScaler',
     'StandardScaler()', 'NUM', False),
    ('sk_minmax_scaler', 'scikit-learn', 'from sklearn.preprocessing import MinMaxScaler',
     'MinMaxScaler()', 'NUM', False),
    ('sk_robust_scaler', 'scikit-learn', 'from sklearn.preprocessing import RobustScaler',
     'RobustScaler()', 'NUM', False),
    ('sk_quantile_transformer', 'scikit-learn',
     'from sklearn.preprocessing import QuantileTransformer',
     'QuantileTransformer(n_quantiles=100, random_state=0)', 'NUM', False),
    ('sk_simple_imputer_mean', 'scikit-learn', 'from sklearn.impute import SimpleImputer',
     "SimpleImputer(strategy='mean')", 'NUM', False),
    ('sk_simple_imputer_median', 'scikit-learn', 'from sklearn.impute import SimpleImputer',
     "SimpleImputer(strategy='median')", 'NUM', False),
    ('sk_knn_imputer', 'scikit-learn', 'from sklearn.impute import KNNImputer',
     'KNNImputer(n_neighbors=5)', 'NUM', False),
    ('sk_iterative_imputer', 'scikit-learn',
     'from sklearn.experimental import enable_iterative_imputer\nfrom sklearn.impute import IterativeImputer',
     'IterativeImputer(random_state=0)', 'NUM', False),
    ('sk_pca', 'scikit-learn', 'from sklearn.decomposition import PCA',
     'PCA(n_components=3, random_state=0)', 'NUMFILLED', False),
    ('sk_kbins', 'scikit-learn', 'from sklearn.preprocessing import KBinsDiscretizer',
     "KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')",
     'NUMFILLED', False),
    ('sk_onehot', 'scikit-learn', 'from sklearn.preprocessing import OneHotEncoder',
     "OneHotEncoder(handle_unknown='ignore', sparse_output=False)", 'CAT', False),
    ('sk_selectkbest', 'scikit-learn',
     'from sklearn.feature_selection import SelectKBest, f_classif',
     'SelectKBest(f_classif, k=3)', 'NUMFILLED', True),
    ('sk_target_encoder', 'scikit-learn', 'from sklearn.preprocessing import TargetEncoder',
     'TargetEncoder(random_state=0)', 'CAT', True),
    ('ce_target_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.TargetEncoder(cols=CAT)', 'CAT', True),
    ('ce_loo_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.LeaveOneOutEncoder(cols=CAT)', 'CAT', True),
    ('ce_catboost_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.CatBoostEncoder(cols=CAT)', 'CAT', True),
    ('ce_james_stein_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.JamesSteinEncoder(cols=CAT)', 'CAT', True),
    ('ce_m_estimate_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.MEstimateEncoder(cols=CAT)', 'CAT', True),
    ('ce_woe_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.WOEEncoder(cols=CAT)', 'CAT', True),
    ('ce_count_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.CountEncoder(cols=CAT)', 'CAT', False),
    ('ce_ordinal_encoder', 'category_encoders', 'import category_encoders as ce',
     'ce.OrdinalEncoder(cols=CAT)', 'CAT', False),
]


def render_temporal(case, n_test):
    cid, lib, imports, code, adm, masked, reason = case
    return T_HEADER.format(IMPORTS=imports, N_TEST=n_test) + code + '\n' + T_FOOTER


def render_tabular(case, pattern, NUM, CAT):
    cid, lib, imports, ctor, cols, uses_y = case
    src = TAB_HEADER.format(IMPORTS=imports, NUM=repr(NUM), CAT=repr(CAT))
    sel = {'NUM': 'NUM', 'NUMFILLED': 'NUM', 'CAT': 'CAT'}[cols]
    if cols == 'NUMFILLED':
        src += "X[NUM] = X[NUM].fillna(0.0)\n"
    yarg = ', y' if uses_y else ''
    ytr = ', y_train' if uses_y else ''
    split = ("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, "
             "random_state=0)\n")
    if pattern == 'A':
        src += f"Z = {ctor}.fit_transform(X[{sel}]{yarg})\n"
        src += "Z = pd.DataFrame(np.asarray(Z, dtype=float), index=X.index).add_prefix('z')\n"
        src += "X = pd.concat([X.drop(columns=CAT), Z], axis=1)\n" if sel == 'CAT' else \
               "X = pd.concat([X.drop(columns=NUM + CAT), Z], axis=1)\n"
        src += split
    else:
        src += split
        src += f"enc = {ctor}\n"
        src += f"Z_train = enc.fit_transform(X_train[{sel}]{ytr})\n"
        src += f"Z_test = enc.transform(X_test[{sel}])\n"
        keep = 'X_train.drop(columns=CAT)' if sel == 'CAT' else 'X_train.drop(columns=NUM + CAT)'
        keep_te = keep.replace('X_train', 'X_test')
        src += (f"X_train = pd.concat([{keep}, pd.DataFrame(np.asarray(Z_train, dtype=float), "
                "index=X_train.index).add_prefix('z')], axis=1)\n")
        src += (f"X_test = pd.concat([{keep_te}, pd.DataFrame(np.asarray(Z_test, dtype=float), "
                "index=X_test.index).add_prefix('z')], axis=1)\n")
    # remaining raw numeric columns (if any) are filled with a constant; the model needs no NaN
    src += ("X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0.0)\n"
            "X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)\n"
            "clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)\n"
            "acc = clf.score(X_test, y_test)\n")
    return src


def write_sources(root, meta, n_test=56):
    d = os.path.join(root, 'library')
    os.makedirs(d, exist_ok=True)
    rows = []
    for case in TEMPORAL:
        path = os.path.join(d, f'T__{case[0]}.py')
        open(path, 'w').write(render_temporal(case, n_test))
        rows.append(dict(kind='temporal', case=case[0], library=case[1], expected_admissible=case[4],
                         masked=case[5], reason=case[6], path=path))
    m = meta['german']
    for case in TABULAR:
        for pattern in ('A', 'B'):
            path = os.path.join(d, f'K__{case[0]}__{pattern}.py')
            open(path, 'w').write(render_tabular(case, pattern, m['num'], m['cat']))
            rows.append(dict(kind='tabular', case=case[0], library=case[1], pattern=pattern,
                             uses_labels=case[5], expected_admissible=(pattern == 'B'),
                             path=path))
    return rows


if __name__ == '__main__':
    import json
    import sys
    meta = json.load(open(sys.argv[2]))
    rows = write_sources(sys.argv[1], meta)
    json.dump(rows, open(os.path.join(sys.argv[1], 'library_index.json'), 'w'), indent=1)
    print(len(rows), 'library sources written')
