import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

# training the linear regression model on the whole dataset

from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X, y)

# training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# visualising the linear regression results
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X), color= 'blue')    # plotting the linear regression line for the X values and the predicted Salary i.e., lin_reg
plt.title('Truth or Bluff (Linear Regression )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color= 'blue')    # plotting the linear regression line for the X values and the predicted Salary i.e., lin_reg
plt.title('Truth or Bluff (Polynomial Regression )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)                   # choosing each point with a diff of 0.1
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color= 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression Smooth)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting a new result with linear regression
print(lin_reg.predict([[6.5]]))

# predicting a new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


