import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
df = pd.read_csv('data.csv')
NUM = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']
CAT = ['A1', 'A3', 'A4', 'A6', 'A7', 'A9', 'A10', 'A12', 'A14', 'A15', 'A17', 'A19', 'A20']
df[CAT] = df[CAT].fillna('missing')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
enc = ce.TargetEncoder(cols=CAT)
Z_train = enc.fit_transform(X_train[CAT], y_train)
Z_test = enc.transform(X_test[CAT])
X_train = pd.concat([X_train.drop(columns=CAT), pd.DataFrame(np.asarray(Z_train, dtype=float), index=X_train.index).add_prefix('z')], axis=1)
X_test = pd.concat([X_test.drop(columns=CAT), pd.DataFrame(np.asarray(Z_test, dtype=float), index=X_test.index).add_prefix('z')], axis=1)
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0.0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)
clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
