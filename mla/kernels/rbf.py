import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist

class RBF(BaseKernel):
    def __init__(self,l=1,sigma=1):
        self.l =l
        self.sigma = sigma

        self.to_precompute = set(['distance'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'distance' not in kwargs :
            dist = cdist(x,y)
        else : 
            dist = kwargs['distance']
        
        return self.sigma**2 * np.exp(-0.5*dist**2/self.l**2 ) 