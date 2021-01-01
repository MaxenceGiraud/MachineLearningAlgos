import numpy as np
from .activation import BaseActivation

class Tanh(BaseActivation):
    def f(self,X):
        return np.tanh(X) # (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    
    def deriv(self,X):
        return 1-np.tanh(X)**2