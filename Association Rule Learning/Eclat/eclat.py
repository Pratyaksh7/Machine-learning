# Eclat -> simplified version of apriori (Main element is Support)
# support means Most frequent combination of products that are bought
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)  # now this file include the 1st row too
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the Eclat model on the dataset
from apyori import apriori as apr
rules = apr(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
# min_support -> 3 common transactions a day * 7(for a week) / 7501
# min_confidence -> 20 %
# min_length and max_length -> deal (buy 1 get 1 free)


# VISUALISING THE RESULTS
# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
for i in results:
    print(i)

# Putting the results well organized into a Pandas Dataframe
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Displaying the results sorted by descending lifts

print(resultsinDataFrame.nlargest(n=10, columns='Support'))
