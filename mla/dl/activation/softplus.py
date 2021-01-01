import numpy as np
from .activation import BaseActivation

class Softplus(BaseActivation):
    def f(self,X):
        return np.log(1+np.exp(X))
    
    def deriv(self,X):
        return 1 / (1 + np.exp(-X))