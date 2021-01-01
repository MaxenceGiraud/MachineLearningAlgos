import numpy as np
from .activation import BaseActivation

class LogActivation(BaseActivation):
    def __init__(self,activation):
        self.activation = activation

    def f(self,X):
        return np.log(self.activation.f(X))
    
    def deriv(self,X):
        return self.activation.deriv(X) / self.activation.f(X)