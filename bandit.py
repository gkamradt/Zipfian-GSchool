import numpy as np


class Bandits(object):
    '''
    This class represent n bandit machines.
    '''

    def __init__(self, p_array):
        '''
        INPUT: list of floats
        OUTPUT: None
        Takes a list of probabilities (probability of conversion) and
        initializes the bandit machines.
        '''
        self.p_array = p_array
        self.optimal = np.argmax(p_array)
        
    def pull(self, i):
        '''
        INPUT: 
        OUTPUT: Bool
        Returns True if the choosing the ith arm led to a conversion.
        '''
        return np.random.random() < self.p_array[i]
    
    def __len__(self):
        return len(self.p_array)

def regret(probabilities, choices):
    '''
    INPUT: array of floats (0 to 1), array of ints
    OUTPUT: array of floats
    Take an array of the true probabilities for each machine and an
    array of the indices of the machine played at each round.
    Return an array giving the total regret after each round.
    '''
    p_opt = np.max(probabilities)
    return np.cumsum(p_opt - probabilities[choices])