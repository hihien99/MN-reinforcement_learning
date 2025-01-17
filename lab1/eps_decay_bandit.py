import numpy as np


class eps_decay_bandit:
    '''
    epsilon-greedy k-bandit class
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    runs: number of runs (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, beta, runs, mu='random', decay=True):
        # Initialization of one instance of the class
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Update decay parameter
        self.beta = beta
        # Number of iterations
        self.runs = runs
        # use decay 
        self.dacy = decay
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(runs)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)


    def update_eps(self):
        eps = 1 /(1 + self.beta * self.n)
        return eps
      
    def pull(self):
        # Pull an arm (epsilon-greedy policy)    
        # Generate random number in [0,1]
        p = np.random.rand()

        # update eps 
        self.eps = self.update_eps()
        
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)

        # Random generation of reward            
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        # Run a simulation for one bandit realization
        for i in range(self.runs):
            self.pull()
            self.reward[i] = self.mean_reward
            
    
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.runs)
        self.k_reward = np.zeros(self.k)