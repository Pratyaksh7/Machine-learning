import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Linear regreesion is used to find the correlation between the X and Y
# It is used on a continuous data/ used to predict real continuous value like Salary

# Training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')            # we have to scatter the training values
plt.plot(X_train, regressor.predict(X_train), color= 'blue')  # the regression line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')            # we have to scatter the test values
plt.plot(X_train, regressor.predict(X_train), color= 'blue')  # the regression line remains the same as unique logic is applied
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""
To predict the salary for a particular years of experience 
Pass the years of experience in the form of 2D Array in regressor.predict()    
"""
print(regressor.predict([[20]]))
# output is [213031.60168521] which is the expected salary for experience of 20 years

print(regressor.coef_) # gives the coefficient i.e., m (slope)
print(regressor.intercept_) # gives the intercept i.e., c (constant) ,  y = mx+c
