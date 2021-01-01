import numpy as np
from .activation import BaseActivation

class Relu(BaseActivation):
    def f(self,X):
        return np.maximum(X,0)
    
    def deriv(self,X):
        return np.where(X>=0,1,0)