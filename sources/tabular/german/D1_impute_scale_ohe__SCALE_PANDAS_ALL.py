import numpy as np
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
NUM = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']
CAT = ['A1', 'A3', 'A4', 'A6', 'A7', 'A9', 'A10', 'A12', 'A14', 'A15', 'A17', 'A19', 'A20']
df[CAT] = df[CAT].fillna('missing')
X = df.drop(columns=['label'])
y = df['label']
X[NUM] = (X[NUM] - X[NUM].mean()) / X[NUM].std()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
imp = SimpleImputer(strategy='median').fit(X_train[NUM])
X_train[NUM] = imp.transform(X_train[NUM])
X_test[NUM] = imp.transform(X_test[NUM])
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[CAT])
X_train = np.hstack([X_train[NUM].to_numpy(), ohe.transform(X_train[CAT])])
X_test = np.hstack([X_test[NUM].to_numpy(), ohe.transform(X_test[CAT])])
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
