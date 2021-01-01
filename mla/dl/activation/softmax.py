import numpy as np
from .activation import BaseActivation

class Softmax(BaseActivation):
    def f(self,X):
        exps = np.exp(X- X.max(axis=1).reshape(-1,1))
        return exps / np.sum(exps,axis=1).reshape(-1,1)
    
    def deriv(self,X):
        return self.f(X) *( 1-self.f(X))