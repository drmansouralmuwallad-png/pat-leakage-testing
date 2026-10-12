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
NUM = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CAT = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
df[CAT] = df[CAT].fillna('missing')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
imp = SimpleImputer(strategy='median').fit(X_train[NUM])
X_train[NUM] = imp.transform(X_train[NUM])
X_test[NUM] = imp.transform(X_test[NUM])
sc = StandardScaler().fit(X_train[NUM])
X_train[NUM] = sc.transform(X_train[NUM])
X_test[NUM] = sc.transform(X_test[NUM])
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[CAT])
X_train = np.hstack([X_train[NUM].to_numpy(), ohe.transform(X_train[CAT])])
X_test = np.hstack([X_test[NUM].to_numpy(), ohe.transform(X_test[CAT])])
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
