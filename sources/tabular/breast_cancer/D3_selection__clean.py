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
NUM = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error', 'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
CAT = []
X = df.drop(columns=['label'])
y = df['label']
X = X[NUM].fillna(0.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sel = SelectKBest(f_classif, k=15).fit(X_train, y_train)
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = clf.score(X_test, y_test)
