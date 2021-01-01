import numpy as np
from .activation import BaseActivation

class Sigmoid(BaseActivation):
    def f(self,X):
        return 1 / (1 + np.exp(-X))
    
    def deriv(self,X):
        return np.exp(-X) / (1+np.exp(-X))**2