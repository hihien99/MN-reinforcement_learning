# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:44:46 2021

@author: Kieffer
"""

import random
import numpy as np 
import matplotlib.pyplot as plt

class cliff_walking:
    #Initialization    
    def __init__(self,gamma):
        # Set up the initial environment
        self.num_rows = 4
        self.num_cols = 12

        self.action_space = [(1, 0), (-1, 0), (0, 1), (0, -1)] # Right, Left, Up, Down
        self.num_actions = len(self.action_space)

        self.state_space = [(i,j) for i in range(self.num_cols) for j in range(self.num_rows)]
        self.initial_state = (0,0)
        self.final_state = (11,0)
        self.cliff = [(i,0) for i in range(1,11)]
        
        self.current_state = self.initial_state
        self.finished = False

        
    # Reward evaluation function
    def reward(self,s_t,a_t):
        if s_t == self.final_state:
            return 0

        # Apply action
        candidate_state = (s_t[0]+a_t[0],s_t[1]+a_t[1])
    
        # checks whether action is valid
        if candidate_state in self.state_space:
            if candidate_state in self.cliff:
                return -100
            else:
                return -1
        else:
            return -1

    # Transition probability evaluation
    def transition_prob(self,s_t,a_t,s_t1):
    # Case of final state
        if s_t == self.final_state:
            if s_t1 == self.final_state:
                return 1
            else:
                return 0
    
        # Apply action
        candidate_state = (s_t[0]+a_t[0],s_t[1]+a_t[1])

        # Moves to cliff?
        if candidate_state in self.cliff:
            if s_t1 == self.initial_state:
                return 1
            else:
                return 0

    
        # Checks whether action is valid
        if candidate_state in self.state_space:
            if candidate_state == s_t1:
                return 1
        else:
            if s_t == s_t1:
                return 1
        return 0


    # Reset
    def reset(self):
        self.current_state = self.initial_state
        self.finished = False
        
        return self.current_state,self.finished

        
    # Apply an action from the current state
    def step(self,a_t):
        if self.finished:
            return self.current_state,0,self.finished
            
        p=[]
        for s in range(0,len(self.state_space)):
            s_t1 = self.state_space[s]
            
            # Evaluates transition probability
            p.append(self.transition_prob(self.current_state,a_t,s_t1))
            
        s_t1 = random.choices(self.state_space,p)
        
        r = self.reward(self.current_state,a_t)
        
        self.current_state = s_t1[0]
        self.finished = (self.current_state == self.final_state) | (r==-100)

        return s_t1[0],r,self.finished
                
    # Rendering function
    def render(self): 
        plt.clf()
        
        plt.xlim([-0.5, self.num_cols-0.5])
        plt.ylim([-0.5, self.num_rows-0.5])
    
        for s in range(0,len(self.state_space)):
            state = self.state_space[s]
            if state == self.initial_state:
                plt.text(state[0], state[1], "S", size=15,
                         ha="center", va="center")
            elif state == self.final_state:
                plt.text(state[0], state[1], "G", size=15,
                         ha="center", va="center")
            elif state in self.cliff:
                plt.text(state[0], state[1], "#", size=15,
                         ha="center", va="center")
        
        plt.text(self.current_state[0], self.current_state[1], "R", size=18,
                 ha="center", va="center")
        
        plt.locator_params(axis='x',tight=True, nbins=self.num_cols)
        plt.locator_params(axis='y',tight=True, nbins=self.num_rows)
        
        plt.grid()
        plt.show()

    # Displays the Value function
    def display_value_policy(self,Q):
        
        plt.xlim([-0.5, self.num_cols-0.5])
        plt.ylim([-0.5, self.num_rows-0.5])
        
        for state in self.state_space:
            s_t = env.state_space.index(state)
            a_t = np.argmax(Q[s_t])
            
            action = self.action_space[a_t]
            
            plt.text(state[0], state[1], "{:10.2f}".format(Q[s_t][a_t]), size=8,
                     ha="center", va="center")
            
            plt.arrow(state[0], state[1], action[0]/3, action[1]/3, head_width=0.05, head_length=0.1, fc='r', ec='r')

        plt.show()


# def SARSA(env,alpha=0.1,gamma=0.9, nb_episodes=2000):
#     """
#     Parameters
#     ----------
#     env : considered environment
#     alpha : learning rate
#     gamma : discount factor

#     Returns
#     -------
#     Q-table

#     """
#     # Initialization of the Q-table
#     Q = [[0 for action in env.action_space] for state in env.state_space]
    
#     # initialization of the return
#     acc_reward = np.zeros(nb_episodes)
    
#     eps = 0.01
    
#     for ep in range(nb_episodes):
#         s_t,finished = env.reset()
        
#         # Initial random action selection
# 	# TO FILL

        
#         while finished == False:
# 	# TO FILL

#     plt.plot(acc_reward)
#     plt.show()    
        
#     return Q
    
# Main function

gamma = 0.9
env = cliff_walking(gamma)

env.render()

# Random policy 
for i in range(10):
    action = random.choice(env.action_space)
    s_t1,reward,finished = env.step(action)

    print(action[0])
    print(s_t1,reward,finished)
    
env.render()

Q = SARSA(env)

env.display_value_policy(Q)