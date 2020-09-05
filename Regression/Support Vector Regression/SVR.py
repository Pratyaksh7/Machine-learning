import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

y = y.reshape(len(y), 1)
# reshape("into" num_of_rows,"into" num_of_cols)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()            # scaling In between -3 to +3
sc_y = StandardScaler()
X = sc_X.fit_transform(X)         # data is converted from original values to values b/w -3 to 3
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
    # first 6.5 has to be converted into range b/w -3 to 3 -> so use sc_X.transform()
    # then we predict its value  which is in the range -3 to 3 -> so we use regressor.predict()
    # since being a user we will not understand that Scaling data
    # so we will re-transform that output so that we get the original value of salary -> inverse_transform()

# Visualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results(for higher resolution and smoother curve)

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)                   # choosing each point with a diff of 0.1
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Truth or Bluff (Support Vector Regression Smooth)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()