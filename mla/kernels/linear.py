import numpy as np
from .base_kernel import BaseKernel

class Linear(BaseKernel):
    def __init__(self):
        self.to_precompute = set(['inner_product'])
    
    def _compute_kernel(self,x,y,**kwargs):
        if 'inner_product' not in kwargs :
            prod = x @ y.T
        else : 
            prod = kwargs['inner_product']
        
        return prod