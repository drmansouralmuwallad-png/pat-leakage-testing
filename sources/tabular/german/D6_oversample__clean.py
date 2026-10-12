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
X = X[NUM].fillna(0.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = pd.concat([X_train, y_train], axis=1)
minority = data[data['label'] == 1]
majority = data[data['label'] == 0]
up = resample(minority, replace=True, n_samples=len(majority), random_state=0)
data = pd.concat([majority, up])
X_train, y_train = data.drop(columns=['label']), data['label']
sc = StandardScaler().fit(X_train[NUM])
X_train[NUM] = sc.transform(X_train[NUM])
X_test[NUM] = sc.transform(X_test[NUM])
X_train = X_train[NUM].to_numpy()
X_test = X_test[NUM].to_numpy()
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
