import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist
from .metrics import chi_metric

class Chi2(BaseKernel):
    def __init__(self,gamma = 1):
        self.gamma = gamma

        self.to_precompute = set(['chi'])
    
    def _compute_kernel(self,x,y,**kwargs):
        if 'chi' not in kwargs :
            chi = cdist(x,y,metric=chi_metric)
        else : 
            chi = kwargs['chi']
        
        return np.exp(-self.gamma * chi) 