import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from feature_engine.timeseries.forecasting import LagFeatures
df = pd.read_csv('series.csv')
N_TEST = 56
y = df['y']
f = LagFeatures(variables=['y'], periods=1).fit_transform(df[['y']])['y_lag_1']
feat = pd.DataFrame({'f': np.asarray(f, dtype=float).ravel()}, index=df.index).shift(1)
feat = feat.iloc[60:].fillna(0.0)
target = y.loc[feat.index].ffill()
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test)
