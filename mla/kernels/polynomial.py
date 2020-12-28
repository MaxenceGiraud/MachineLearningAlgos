import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist

class Polynomial(BaseKernel):
    def __init__(self,degree=2,gamma=0.1,coef0=1):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.to_precompute = set(['scalar_product'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'scalar_product' not in kwargs :
            prod = x @ y.T
        else : 
            prod = kwargs['scalar_product']
        
        return (self.gamma * prod + self.coef0)**self.degree