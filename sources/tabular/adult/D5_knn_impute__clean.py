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
X = X[NUM]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNNImputer(n_neighbors=5).fit(X_train)
X_train = pd.DataFrame(knn.transform(X_train), index=X_train.index, columns=NUM)
X_test = pd.DataFrame(knn.transform(X_test), index=X_test.index, columns=NUM)
sc = StandardScaler().fit(X_train[NUM])
X_train[NUM] = sc.transform(X_train[NUM])
X_test[NUM] = sc.transform(X_test[NUM])
X_train = X_train[NUM].to_numpy()
X_test = X_test[NUM].to_numpy()
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
