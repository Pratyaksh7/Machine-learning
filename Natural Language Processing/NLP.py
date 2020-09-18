# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer   #  -> Since love and loved means the same so to convert from loved to love , use this lib.import

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])      # from dataset remove everything which is not alphabets
    review = review.lower()   # then change all the upper case letters to uppercase
    review = review.split()   # then split to create a list of words from each review

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    # for each word in each review if the word is not like(a, an, the...etc), only then do the stemming(loved->love)
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# Creating the bag of words model      -> Sentiment Analysis
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)   # takes only the words with more frequency
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
print(len(X[0]))  #  -> 1566 words are in corpus

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))   # Accuracy is 0.73 (73%)

# Text Cleaning can be done more efficiently for more accuracy