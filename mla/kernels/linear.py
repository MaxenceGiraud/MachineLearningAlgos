import numpy as np
from .base_kernel import BaseKernel

class Linear(BaseKernel):
    def __init__(self):
        self.to_precompute = set(['scalar_product'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'scalar_product' not in kwargs :
            prod = x @ y.T
        else : 
            prod = kwargs['scalar_product']
        
        return prod