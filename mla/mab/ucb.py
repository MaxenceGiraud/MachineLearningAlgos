from .randmax import randmax
import numpy as np

class UCB:
    """UCB1 with parameter alpha
    
    Parameters
    ----------
    nbArms :int,
        Number of arms of bandit
    alpha : float,
    """
    def __init__(self, nbArms,alpha=2):
        self.nbArms = nbArms
        self.clear()
        self.Best = 0
        self.alpha = alpha

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.arm_means = np.zeros(self.nbArms)
        self.t = 0
    
    def chooseArmToPlay(self):
        if self.t < self.nbArms :
            return self.t
        else :
            return randmax(self.cumRewards/self.nbDraws + np.sqrt(self.alpha * np.log(self.t)/self.nbDraws))

    def receiveReward(self, arm, reward):
        self.t  += 1 
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
    
    def name(self):
        return "UCB"