import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist

class Laplacian(BaseKernel):
    def __init__(self,gamma = 1):
        self.gamma = gamma

        self.to_precompute = set(['distance_manhattan'])
    
    def _compute_kernel(self,x,y,**kwargs):
        if 'distance_manhattan' not in kwargs :
            dist = cdist(x,y,metric='cityblock')
        else : 
            dist = kwargs['distance_manhattan']
        
        return np.exp(-self.gamma * dist ) 