import numpy as np
from .activation import BaseActivation

class LeakyRelu(BaseActivation):
    def __init__(self,alpha):
        self.alpha = alpha
    def f(self,X):
        return np.where(X>=0,X,self.alpha*X)
    
    def deriv(self,X):
        return np.where(X>=0,1,self.alpha)