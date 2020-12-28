import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist

class Laplacian(BaseKernel):
    def __init__(self,gamma = 1):
        self.gamma = gamma

        self.to_precompute = set(['distance_manhattan'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'distance_manhattan' not in kwargs :
            dist = cdist(x,y,metric='cityblock')
        else : 
            dist = kwargs['distance_manhattan']
        
        return np.exp(-self.gamma * dist ) 