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
feat['roll_mean'] = y_feat.shift(1).rolling(M).mean()
feat['roll_std'] = y_feat.shift(1).rolling(M).std()
feat = feat.iloc[3 * M:]
target = y.loc[feat.index]
feat = pd.DataFrame(StandardScaler().fit_transform(feat), index=feat.index, columns=feat.columns)
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
mu, sd = y_train.mean(), y_train.std()
y_train = (y_train - mu) / sd
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test) * sd + mu
rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
