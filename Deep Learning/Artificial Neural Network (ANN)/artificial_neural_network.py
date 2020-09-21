# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
 # Label Encoding the Gender Column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

 # One Hot Encoding the "Geography" Column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling (Its very important step in Deep learning)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)     # here we apply feature scaling on every column
X_test = sc.transform(X_test)

# Part2 - Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # relu -> rectifier

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # 6 neurons with each neuron uses rectifier activation function

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # 1 output layer, and sigmoid represent Binary value

# Part3 - Training the ANN
 # Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                            # loss = 'categorical_crossentropy', if there are more categories

 # Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Part4 - Making the prediction and Evaluating the model
 # Predicting the result of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

 # Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)   # so that the result is in the form of true(1) or false(0) and not in probability


print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

 # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Accuracy came to be 0.8624