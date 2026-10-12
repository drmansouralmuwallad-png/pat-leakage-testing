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
N_TEST = 84
n_fit = len(df) - N_TEST
y = df['y']
y_feat = y
feat = pd.DataFrame(index=df.index)
for k in range(1, M + 1):
    feat['lag_%d' % k] = y_feat.shift(k)
feat['lag_2m'] = y_feat.shift(2 * M)
feat['roll_mean'] = y_feat.shift(1).rolling(M, center=True, min_periods=1).mean()
feat['roll_std'] = y_feat.shift(1).rolling(M, center=True, min_periods=2).std()
feat = feat.iloc[3 * M:]
target = y.loc[feat.index]
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
mu, sd = y_train.mean(), y_train.std()
y_train = (y_train - mu) / sd
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test) * sd + mu
rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
