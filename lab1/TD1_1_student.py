# -*- coding: utf-8 -*-
"""
E3A - 456

The content of this program has been largely taken from

https://towardsdatascience.com 

Last update 27/10/2021
"""

# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from eps_bandit import eps_bandit        

# Main part of the program
# Number of arms
k = 10
# Number of runs for each bandit realization
runs = 1000

# Array to store the rewards
eps_1_rewards = np.zeros(runs)

# Number of realizations for the simulation 
realizations = 100

# Run experiments
for i in range(realizations):
    # Initialize bandits
    eps_1 = eps_bandit(k, 0.1, runs)
    
    # Run experiments
    eps_1.run()
    
    # Update long-term averages among episodes
    eps_1_rewards = eps_1_rewards + (
        eps_1.reward - eps_1_rewards) / (i + 1)

plt.figure(figsize=(14,8))
plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
plt.legend(bbox_to_anchor=(0.6, 0.5))
plt.xlabel("Runs")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon$-greedy and greedy rewards after " + str(realizations) 
    + " Realizations")
plt.show()
