import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3] )] , remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# split the test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# training the Multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)), axis=1))
# this is used to concatenate the y_pred -> predicted vaues to the y_test -> actual values rowise instead of columnwise

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000,
# Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
#     example:-
                # regressor.predict([[1, 0, 0, 160000, 130000, 300000]])
                # output-> [181566.92]
