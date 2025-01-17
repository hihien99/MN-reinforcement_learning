from eps_bandit import *
from eps_decay_bandit import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def man1(k, runs, realizations, eps):
    """
    k : the number of arms
    runs: number of runs for each bandit realization
    realizations: number of performed experiments
    eps: epsilon for algorithm
    
    return:
    eps_1_rewards: long-term average rewards
    eps_0_rewards: long-term average rewards
    eps_01_rewards: long-term average rewards
    """
    eps_1_rewards = np.zeros(runs)
    eps_0_rewards = np.zeros(runs)
    eps_01_rewards = np.zeros(runs)

    # run experiments
    for i in range(realizations):
        # initialize bandit
        eps_1 = eps_bandit(k=k, eps=eps, runs=runs)
        eps_0 = eps_bandit(k=k, eps=0.0, runs=runs, mu=eps_1.mu.copy())
        eps_01 = eps_bandit(k=k, eps=0.01, runs=runs, mu=eps_1.mu.copy())


        # run experiments
        eps_1.run()
        eps_0.run()
        eps_01.run()

        # Update long-term averages among episodes
        eps_1_rewards = eps_1_rewards + (
            eps_1.reward - eps_1_rewards) / (i + 1)
        eps_0_rewards = eps_0_rewards + (
            eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (
            eps_01.reward - eps_01_rewards) / (i + 1)
        rewards = (eps_0_rewards, eps_1_rewards, eps_01_rewards)
                
    return eps_1_rewards, eps_0_rewards, eps_01_rewards, rewards


def man2(k, runs, realizations, eps):
    """
    k : the number of arms
    runs: number of runs for each bandit realization
    realizations: number of performed experiments
    eps: epsilon for algorithm
    
    return:
    
    """
    eps_1_rewards = np.zeros(runs)
    eps_0_rewards = np.zeros(runs)
    eps_01_rewards = np.zeros(runs)

    # array to store the average action selection among realizations  
    eps_1_selection = np.zeros(k)
    eps_0_selection = np.zeros(k)
    eps_01_selection = np.zeros(k)

    # run experiments
    for i in range(realizations):
        # initialize bandit
        eps_1 = eps_bandit(k=k, eps=0.0, runs=runs, mu="sequence")
        eps_0 = eps_bandit(k=k, eps=0.0, runs=runs, mu=eps_1.mu.copy())
        eps_01 = eps_bandit(k=k, eps=0.01, runs=runs, mu=eps_1.mu.copy())

        # run experiments
        eps_1.run()
        eps_0.run()
        eps_01.run()

        # Update long-term averages among episodes
        eps_1_rewards = eps_1_rewards + (
            eps_1.reward - eps_1_rewards) / (i + 1)
        eps_0_rewards = eps_0_rewards + (
            eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (
            eps_01.reward - eps_01_rewards) / (i + 1)
        rewards = (eps_1_rewards, eps_0_rewards, eps_01_rewards)
        
        # average actions per realization 
        eps_1_selection = eps_1_selection + (
            eps_1.k_n - eps_1_selection) /(i + 1)
        eps_0_selection = eps_0_selection + (
            eps_0.k_n - eps_0_selection) /(i + 1)
        eps_01_selection = eps_01_selection + (
            eps_01.k_n - eps_01_selection) /(i + 1)
        selections = (eps_0_selection, eps_1_selection, eps_01_selection)
                
    return rewards, selections


def man3_plot(n, beta):
    """
    n: number of run for a realization
    beta: exploration rate beta
    """

    # Generate data for each beta value
    n = np.arange(n)  # Steps from 0 to n_steps - 1

    plt.figure(figsize=(10, 6))

    for bet in beta:
        eps_values = 1 / (1 + bet * n)
        plt.plot(n, eps_values, label=f"beta = {bet}")

    # Plot settings
    plt.title("Evolution of e as a function of n for different beta Values", fontsize=14)
    plt.xlabel("Steps (n)", fontsize=12)
    plt.ylabel("Exploration Rate (beta)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def man3(k, runs, realizations, eps, beta):
    """
    k : the number of arms
    runs: number of runs for each bandit realization
    realizations: number of performed experiments
    eps: epsilon for algorithm
    
    return:
    eps_1_rewards: long-term average rewards
    eps_0_rewards: long-term average rewards
    eps_01_rewards: long-term average rewards
    """
    eps_1_rewards = np.zeros(runs)
    eps_0_rewards = np.zeros(runs)
    eps_01_rewards = np.zeros(runs)

    eps_1_decay_rewards = np.zeros(runs)
    eps_0_decay_rewards = np.zeros(runs)
    eps_01_decay_rewards = np.zeros(runs)

    rewards_decays = []

    # run experiments
    for i in range(realizations):
        # initialize bandit
        eps_1 = eps_bandit(k=k, eps=eps, runs=runs)
        eps_0 = eps_bandit(k=k, eps=0.0, runs=runs, mu=eps_1.mu.copy())
        eps_01 = eps_bandit(k=k, eps=0.01, runs=runs, mu=eps_1.mu.copy())

        # run experiments
        eps_1.run()
        eps_0.run()
        eps_01.run()

        # Update long-term averages among episodes
        eps_1_rewards = eps_1_rewards + (
            eps_1.reward - eps_1_rewards) / (i + 1)
        eps_0_rewards = eps_0_rewards + (
            eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (
            eps_01.reward - eps_01_rewards) / (i + 1)
        rewards = (eps_0_rewards, eps_1_rewards, eps_01_rewards)

        for j, bet in enumerate(beta):
            eps_1_decay = eps_decay_bandit(k=k, eps=0.0, beta=bet, runs=runs)
            eps_0_decay = eps_decay_bandit(k=k, eps=0.0, beta=bet, runs=runs, mu=eps_1.mu.copy())
            eps_01_decay = eps_decay_bandit(k=k, eps=0.01, beta=bet, runs=runs, mu=eps_1.mu.copy())

            eps_1_decay.run()
            eps_0_decay.run()
            eps_01_decay.run()

            eps_1_decay_rewards = eps_1_decay_rewards + (
                eps_1_decay.reward - eps_1_decay_rewards) / (i + 1)
            eps_0_decay_rewards = eps_0_decay_rewards + (
                eps_0_decay.reward - eps_0_decay_rewards) / (i + 1)
            eps_01_decay_rewards = eps_01_decay_rewards + (
                eps_01_decay.reward - eps_01_decay_rewards) / (i + 1)
            rewards_decay = (eps_1_decay_rewards, eps_0_decay_rewards, eps_01_decay_rewards)
            rewards_decays.append(rewards_decay)
    
    return eps_1_rewards, eps_0_rewards, eps_01_rewards, rewards, rewards_decays
    

if __name__ == "__main__":

    k = 10
    runs = 1000
    eps = 0.1

    # manipulation 1
    # eps_1_real1, eps_0_real1, eps_01_real1 = man1(k=k, runs=runs, realizations=10, eps=eps)
    # eps_1_real2, eps_0_real2, eps_01_real2 = man1(k=k, runs=runs, realizations=100, eps=eps)
    # eps_1_real3, eps_0_real3, eps_01_real3 = man1(k=k, runs=runs, realizations=1000, eps=eps)
    
    # plt.figure(figsize=(14,8))
    # plt.plot(eps_1_real1, label="$arm1, realization=10$")
    # plt.plot(eps_1_real2, label="$arm1, realization=100$")
    # plt.plot(eps_1_real3, label="$arm1, realization=1000$")

    # plt.plot(eps_0_real1, label="$arm0, realization=10$")
    # plt.plot(eps_0_real2, label="$arm0, realization=100$")
    # plt.plot(eps_0_real3, label="$arm0, realization=1000$")

    # plt.plot(eps_01_real1, label="$arm01, realization=10$")
    # plt.plot(eps_01_real2, label="$arm01, realization=100$")
    # plt.plot(eps_01_real3, label="$arm01, realization=1000$")
    
    # plt.legend(bbox_to_anchor=(0.6, 0.5))
    # plt.xlabel("Runs")
    # plt.ylabel("Average Reward")
    # plt.title("Average $\epsilon$-greedy and greedy rewards with different realizations")
    # plt.show()


    # manipulation 2 
    # rewards_man2, selection_man2 = man2(k=k, runs=runs, realizations=1000, eps=eps)
    # eps_1_rewards, eps_0_rewards, eps_01_rewards, rewards = man1(k=k, runs=runs, realizations=1000, eps=eps)

    # print("performance manipulation 1: ", rewards[0].shape)
    # print("performance manipulation 2: ", len(rewards_man2))
    # print("selection in manipulation 2: ", selection_man2, selection_man2[0].shape)

    # get the chart
    # bins = np.linspace(0, k-1, k)
    # plt.figure(figsize=(12, 8))
    # plt.bar(bins, selection_man2[0], width = 0.33, color="b", label="$\epsilon=0$")
    # plt.bar(bins+0.33, selection_man2[2], width = 0.33, color="g", label="$\epsilon=0.01$")
    # plt.bar(bins+0.66, selection_man2[1], width = 0.33, color="r", label="$\epsilon=0.1$")
    # plt.legend(bbox_to_anchor=(1.2, 0.5))
    # plt.xlim([0,k])
    # plt.title("Actions Selected by Each Algorithm")
    # plt.xlabel("Action")
    # plt.ylabel("Number of Actions Taken")
    # plt.show()

    # get the table
    # opt_per = np.array([selection_man2[0], selection_man2[2], selection_man2[1]]) / runs * 100
    # df = pd.DataFrame(opt_per,
    #                   index=['$\epsilon=0$', '$\epsilon=0.01$', "$\epsilon=0.1$"],
    #                   columns=["a = " + str(x) for x in range(0, k)])
    # print("Percentage of actions selected: " + "\n", df)


    # manipulation 3
    n = 100
    beta = [0.01, 0.05, 0.1]  # Different values of beta
    # man3_plot(n=n, beta=beta)

    eps_1_rewards, eps_0_rewards, eps_01_rewards, rewards, rewards_decays = man3(k=k, runs=runs, realizations=100, eps=eps, beta=beta)
    plt.figure(figsize=(14,8))
    plt.plot(eps_1_rewards, label="$bandit1, without_decay$")
    plt.plot(eps_0_rewards, label="$bandit0, without_decay$")
    plt.plot(eps_01_rewards, label="$bandit01, without_decay$")

    plt.plot(rewards_decays[0][0], label="$abandit1, decay_beta=0.01$")
    plt.plot(rewards_decays[0][1], label="$abandit0, decay_beta=0.01$")
    plt.plot(rewards_decays[0][2], label="$abandit01, decay_beta=0.01$")

    plt.plot(rewards_decays[1][0], label="$abandit1, decay_beta=0.01$")
    plt.plot(rewards_decays[1][1], label="$abandit0, decay_beta=0.05$")
    plt.plot(rewards_decays[1][2], label="$abandit01, decay_beta=0.05$")

    plt.plot(rewards_decays[2][0], label="$abandit1, decay_beta=0.1$")
    plt.plot(rewards_decays[2][1], label="$abandit0, decay_beta=0.1$")
    plt.plot(rewards_decays[2][2], label="$abandit01, decay_beta=0.1$")

    plt.legend(bbox_to_anchor=(0.6, 0.5))
    plt.xlabel("Runs")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon$-greedy and greedy rewards with different realizations")
    plt.show()



