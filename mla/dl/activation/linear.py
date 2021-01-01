import numpy as np
from .activation import BaseActivation

class Linear(BaseActivation):
    def f(self,X):
        return X
    
    def deriv(self,X):
        return np.ones(X.shape)