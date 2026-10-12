import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df = pd.read_csv('sales.csv')
y = df['y']
feat = pd.DataFrame(index=df.index)
for k in range(1, 8):
    feat['lag_%d' % k] = y.shift(k)
feat['roll_mean'] = y.shift(1).rolling(7, center=True, min_periods=1).mean()
feat = feat.iloc[14:]
target = y.loc[feat.index]
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=56, shuffle=False)
model = Ridge(alpha=1.0).fit(X_train, y_train)
