# thompson Sampling :- uses random values
# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing thompson sampling
import random
N = 500
d = 10
ads_selected = []
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # from the algorithm
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i     # because we have to select the ad have max value, so i is the index of that ad

    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1

    total_reward += reward

# visualizing the results: Histogram
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('number of times each ad was selected')
plt.show()
