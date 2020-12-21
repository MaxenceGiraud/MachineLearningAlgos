import numpy as np
from .base_kernel import BaseKernel

class RBF(BaseKernel):
    def __init__(self,l=1,sigma=1):
        self.l =l
        self.sigma = sigma
    
    def f(self,x,y):
        return self.sigma**2 * np.exp(-0.5*np.sum((x-y)**2)/self.l**2 ) 
