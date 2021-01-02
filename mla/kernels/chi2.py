import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist
from .utils import chi_metric

class Chi2(BaseKernel):
    def __init__(self,gamma = 1):
        self.gamma = gamma

        self.to_precompute = set(['chi'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'chi' not in kwargs :
            chi = cdist(x,y,metric=chi_metric)
        else : 
            chi = kwargs['chi']
        
        return np.exp(-self.gamma * chi) 