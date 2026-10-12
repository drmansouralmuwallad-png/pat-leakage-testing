import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter

df = pd.read_csv('series.csv')
M = 7
N_TEST = 56
n_fit = len(df) - N_TEST
y = df['y']
y_feat = y
feat = pd.DataFrame(index=df.index)
for k in range(1, M + 1):
    feat['lag_%d' % k] = y_feat.shift(k)
feat['lag_2m'] = y_feat.shift(2 * M)
feat['phase'] = np.arange(len(df)) % M
dec = seasonal_decompose(y_feat, period=M, two_sided=False)
feat['trend'] = dec.trend.shift(1)
thr = y_feat.iloc[:n_fit].quantile(0.9)
feat['peak_1'] = (y_feat.shift(1) > thr).astype(float)
feat['peak_m'] = (y_feat.shift(M) > thr).astype(float)
feat = feat.iloc[3 * M:]
target = y.loc[feat.index]
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
prof = y_train.groupby(X_train['phase']).mean()
X_train['prof'] = X_train['phase'].map(prof)
X_test['prof'] = X_test['phase'].map(prof)
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test)
rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
