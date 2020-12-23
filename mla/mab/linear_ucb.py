import numpy as np
from .randmax import randmax
from .base_mab import BaseMAB

class LinUCB(BaseMAB):
    """Linear UCB strategy

    Parameters
    ----------
    X : array of shape (K,d),
        Feature matrix witk = Number of arms and d the number of dimensons
    sigma : float,
        aa
    reg : float,
        regularization parameters
    delta ; float,
        aa
    threshold : None or function,
        If None default threshold is selected.
    """
    def __init__(self, X,sigma=0.5,reg=1,delta=0.05,threshold = None):
        self.features = X
        (self.nbArms,self.dimension) = np.shape(X)
        self.reg = reg
        self.clear()

        self.delta = delta
        self.sigma = sigma

        if threshold == None :
            self.threshold = lambda t,delta : self.sigma*np.sqrt(10*np.log(1+t/(self.nbArms*self.reg)))
        else :
            self.threshold = threshold

    def clear(self):
        self.t = 0

        # initialize the design matrix, its inverse, 
        # the vector containing the sum of r_s*x_s and the least squares estimate
        self.Design = self.reg*np.eye(self.dimension)
        self.DesignInv = (1/self.reg)*np.eye(self.dimension)
        self.Vector = np.zeros((self.dimension,1))
        self.thetaLS = np.zeros((self.dimension,1)) # regularized least-squares estimate
    
    def chooseArmToPlay(self):
        # compute the vector of estimated means  
        muhat = (self.features @ self.thetaLS).flatten() + np.diag(np.sqrt(self.features @ self.DesignInv @ self.features.T))*self.threshold(self.t,self.delta)
        # select the arm with largest estimated mean 
        return randmax(muhat)


    def receiveReward(self,arm,reward):
        self.t += 1
        x = self.features[arm,:].reshape((self.dimension,1))
        self.Design =  self.Design +  x @ x.T
        self.Vector = self.Vector + reward*x
        y = self.DesignInv @ x
        # online update of the inverse of the design matrix
        self.DesignInv -=  (1/(1+x.T@y )) * y@y.T
        # update of the least squares estimate 
        self.thetaLS = self.DesignInv @ self.Vector 