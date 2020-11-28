from .randmax import randmax
import numpy as np

class ETC:
    """Explore-Then-Commit strategy for two arms"""
    def __init__(self, nbArms,Horizon,c=1/2):
        self.nbArms = 2
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
        if self.Explore :
            return self.t % self.nbArms
        else :
            return self.Best

    def receiveReward(self, arm, reward):
        self.t  += 1 
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

        if self.Explore :
            arms_mean = self.cumRewards / self.nbDraws
            if np.abs(arms_mean[0]-arms_mean[1]) > np.sqrt( self.c * np.log(self.T/self.t) / self.t):
                self.Best = randmax(arms_mean)
                self.Explore = False
    
    def name(self):
        return "ETC"