# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[: , :-1].values               # independent variable -> features
y = dataset.iloc[: , -1].values                # dependent variable -> usually what we have to find/predict

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[: , 1:3])

# ENCODING CATEGORICAL DATA
# Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

                 # Hint ->
                        # we use OneHotEncoding when have to divide in multiple categories
                        # we use LabelEncoding when we have 2 categories

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
    # since features just belongs to X
    # so we will only standardize X including X_test and X_train
    # values varies between -3 to 3

X_train[: , 3:] = sc.fit_transform(X_train[: , 3:])
X_test[: , 3:] = sc.transform(X_test[: , 3:]) # we donot fit the X_test because its already been trained and hence fitted

print(X_train)
print(X_test)