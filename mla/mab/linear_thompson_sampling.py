import numpy as np
from .randmax import randmax
from .base_mab import BaseMAB

class LinTS(BaseMAB):
    """Linear Thompson Sampling strategy

    Parameters
    ----------
    X : array of shape (K,d),
        Feature matrix witk = Number of arms and d the number of dimensons
    k : float,
        aa
    reg : float,
        regularization parameters
    """
    def __init__(self,X,k=0.25,reg=1):
        # the algorithms is fed with the known matrix of features X of shape (K,d) and the regularization parameter 'reg'

        self.features = X
        (self.nbArms,self.dimension) = np.shape(X)

        self.theta = np.zeros(self.nbArms)  
        self.k = k #prior variance          
        self.v = np.sqrt(reg * self.k**2) # posterior variance param
        self.reg = reg 

        self.clear()
    
    def clear(self):
        # initialize the design matrix, its inverse, 
        # the vector containing the sum of r_s*x_s and the least squares estimate
        self.Design = self.reg*np.eye(self.dimension)
        self.DesignInv = (1/self.reg)*np.eye(self.dimension)
        self.Vector = np.zeros((self.dimension,1))
        self.thetaLS = np.zeros((self.dimension,1)) # regularized least-squares estimate
        self.thetatilda = np.random.multivariate_normal(self.thetaLS.flatten(),self.v**2 * self.DesignInv).reshape((self.dimension,1))

    def chooseArmToPlay(self):
        # compute the vector of estimated means  
        muhat = self.features @ self.thetatilda
        # select the arm with largest estimated mean 
        return randmax(muhat)
    
    def receiveReward(self,arm,reward):
        x = self.features[arm,:].reshape((self.dimension,1))
        self.Design =  self.Design +  x @ x.T
        self.Vector = self.Vector + reward*x
        y = self.DesignInv @ x
        # online update of the inverse of the design matrix
        self.DesignInv -=  (1/(1+x.T@y )) * y@y.T
        # update of the least squares estimate 
        self.thetaLS = self.DesignInv @ self.Vector 
        self.thetatilda = np.random.multivariate_normal(self.thetaLS.flatten(),self.v**2 * self.DesignInv).reshape((self.dimension,1))