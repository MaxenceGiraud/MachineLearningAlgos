from .randmax import randmax
import numpy as np

class ThompsonSampling:
    """Thompson Sampling with Beta(a,b) prior and Bernoulli likelihood

    Parameters
    ----------
    nbArms :int,
        Number of arms of bandit
    alpha : float,
        Added to the first param of the beta distrib
    beta : float,
        Added to the 2nd param of the beta distrib
    """
    def __init__(self, nbArms,alpha = 1,beta= 1):
        self.nbArms = nbArms
        self.clear()
        # Beta distribution parameters
        self.alpha = alpha
        self.beta = beta

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.arm_means = np.zeros(self.nbArms)

    
    def chooseArmToPlay(self):
        return randmax(np.random.beta(self.alpha + self.cumRewards,self.beta + self.nbDraws - self.cumRewards))

    def receiveReward(self, arm, reward):
        # Binarization trick in case reward are not binary
        bin_reward = float(np.random.random()<reward)
        self.cumRewards[arm] = self.cumRewards[arm]+bin_reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
    
    def name(self):
        return 'Thompson Sampling'