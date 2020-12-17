from .randmax import randmax
import numpy as np
from .bandit_env.bandit_tools import klucbBern

class klUCB:
    """kl-UCB
    
    Parameters
    ----------
    nbArms :int,
        Number of arms of bandit
    div : function,
        kl divergence to use
    """
    def __init__(self, nbArms,div=klucbBern):
        self.nbArms = nbArms
        self.clear()
        self.div = div.__name__
        self.f_klucb = div

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.arm_means = np.zeros(self.nbArms)
        self.t = 0
    
    def chooseArmToPlay(self):
        if self.t < self.nbArms :
            return self.t
        else :
            return randmax([self.f_klucb(self.cumRewards[i]/self.nbDraws[i], np.log(self.t) /self.nbDraws[i] ) for i in range(self.nbArms)])

    def receiveReward(self, arm, reward):
        self.t  += 1 
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
    
    def name(self):
        return "klUCB"