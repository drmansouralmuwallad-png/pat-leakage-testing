import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
df = pd.read_csv('data.csv')
NUM = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']
CAT = ['A1', 'A3', 'A4', 'A6', 'A7', 'A9', 'A10', 'A12', 'A14', 'A15', 'A17', 'A19', 'A20']
df[CAT] = df[CAT].fillna('missing')
X = df.drop(columns=['label'])
y = df['label']
Z = RobustScaler().fit_transform(X[NUM])
Z = pd.DataFrame(np.asarray(Z, dtype=float), index=X.index).add_prefix('z')
X = pd.concat([X.drop(columns=NUM + CAT), Z], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0.0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)
clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
