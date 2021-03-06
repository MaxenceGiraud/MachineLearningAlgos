from .randmax import randmax
import numpy as np
from .base_mab import BaseMAB

class ETC(BaseMAB):
    """Explore-Then-Commit strategy for two arms
    
    Parameters
    ----------
    nbArms :int,
        Number of arms of bandit
    Horizon : int,
        Parameter that scale when the best arm is chosen before commiting    
    """
    def __init__(self, nbArms,Horizon,c=1/2):
        self.nbArms = nbArms
        self.T = Horizon
        self.clear()
        self.Explore = True # are we still exploring? 
        self.Best = 0
        self.c = c

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
        self.Explore = True 
    
    def chooseArmToPlay(self):
        if self.Explore : # Exploring
            return self.t % self.nbArms
        else : # Commit
            return self.Best

    def receiveReward(self, arm, reward):
        self.t  += 1 
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

        if self.Explore :
            arms_mean = self.cumRewards / self.nbDraws
            sorted_means = np.sort(arms_mean)
            if np.abs(arms_mean[0]-arms_mean[1]) > np.sqrt( self.c * np.log(self.T/self.t) / self.t):
                self.Best = randmax(arms_mean)
                self.Explore = False