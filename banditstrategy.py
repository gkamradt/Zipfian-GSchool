import scipy.stats as stats
import numpy as np
import random


class BanditStrategy(object):	
    '''
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
		choice_function: accepts a self argument (which gives access to all the variables), and 
						returns and int between 0 and n-1
    methods:
        sample_bandits(n): sample and train on n pulls.
    attributes:
         N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    '''
    
    def __init__(self, bandits, choice_function):
        '''
        INPUT: Bandits, function
        
        Initializes the BanditStrategy given an instance of the Bandits class
        and a choice function.
        '''
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.score = []
        self.choice_function = choice_function

    def sample_bandits(self, n=1):         
        '''
        INPUT: in
        OUTPUT: None 
        Simulate n runds of running the bandit machine.
        '''         
        score = np.zeros(n)
        choices = np.zeros(n)

        # seed the random number generators so you get the same results every
        # time.
        np.random.seed(101)
        random.seed(101)
        
        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = self.choice_function(self)
            
            #sample the chosen bandit
            result = self.bandits.pull(choice)
            
            #update priors and score
            self.wins[choice] += result
            self.trials[choice] += 1
            score[k] = result 
            self.N += 1
            choices[k] = choice
            
        self.score = np.r_[self.score, score]
        self.choices = np.r_[self.choices, choices]

        
def max_mean(self):
    '''
    Pick the bandit with the current best observed proportion of winning.
    Return the index of the winning bandit.
    '''
    # make sure to play each bandit at least once
    if len(self.trials.nonzero()[0]) < len(self.bandits):
        return np.random.randint(0, len(self.bandits))
    return np.argmax(self.wins / (self.trials + 1))
    
def random_choice(self):
    '''
    Pick a bandit uniformly at random.
    Return the index of the winning bandit.
    '''
    return np.random.randint(0, len(self.wins))

def epsilon_greedy(self, epsilon=0.1):
    '''
    Pick a bandit uniformly at random epsilon percent of the time.
    Otherwise pick the bandit with the best observed proportion of winning.
    Return the index of the winning bandit.
    '''
    pass
    r = np.random.random()
    if r <= epsilon:
        return random_choice(self)
    else:
        win_percent = self.wins / self.trials.astype(float)
        return win_percent.argmax(axis=0)

def softmax(self, tau=0.001):
    '''
    Pick an bandit according to the Boltzman Distribution.
    Return the index of the winning bandit.
    '''
    pass
           
    win_percent = (self.wins + 1) / (self.trials.astype(float) + 1)
    
    soft_calc = np.exp(win_percent / tau)
    denom = sum(soft_calc)
  
    return (soft_calc / denom).argmax(axis=0)
        
def ucb1(self):
    '''
    Pick the bandit according to the UCB1 strategy.
    Return the index of the winning bandit.
    '''
    pass

    win_percent = (self.wins) / (self.trials.astype(float))
    ucbl_band = win_percent + np.sqrt((2*np.log(self.N))/self.trials)
    return ucbl_band.argmax(axis=0)
    
def bayesian_bandit(self):
    '''
    Randomly sample from a beta distribution for each bandit and pick the one
    with the largest value.
    Return the index of the winning bandit.
    '''
    pass
    random_beta_sample = stats.beta.rvs(a=1+self.wins, b=1+self.trials-self.wins)
    return random_beta_sample.argmax(axis=0)