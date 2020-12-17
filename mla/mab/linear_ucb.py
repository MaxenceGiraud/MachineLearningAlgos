import numpy as np
from .randmax import randmax

class LinUCB:
    """Linear UCB strategy"""
    def __init__(self, X,sigma=0.5,reg=1,delta=0.05,threshold = None):
        # the algorithms is fed with the known matrix of features (K,d) and the regularization parameter
        self.features = X
        (self.nbArms,self.dimension) = np.shape(X)
        self.reg = reg
        self.clear()

        self.delta = delta
        self.sigma = sigma

        if threshold == None :
            self.threshold = lambda t,delta : self.sigma*np.sqrt(10*np.log(1+t/(self.nbArms*self.reg)))
            #self.threshold = lambda t,delta : self.sigma*np.sqrt(2*np.log(1/delta)+self.nbArms*np.log(1+t/(self.nbArms*self.reg))) + np.sqrt(self.reg) # Theoretical Threshold - Consider norm of theta star =1 and L= 1 as the data is normalized
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
    
    def name(self):
        return "Linear UCB"