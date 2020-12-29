import numpy as np
from .base_kernel import BaseKernel

class Sigmoid(BaseKernel):
    def __init__(self,degree=2,gamma=0.1,r=1):
        self.gamma = gamma
        self.r = r

        self.to_precompute = set(['scalar_product'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'scalar_product' not in kwargs :
            prod = x @ y.T
        else : 
            prod = kwargs['scalar_product']
        
        return np.tanh(self.gamma * prod + self.r)