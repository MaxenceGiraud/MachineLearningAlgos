import numpy as np
from scipy.spatial.distance import cdist
from .base_kernel import BaseKernel

class RQK(BaseKernel):
    '''Rational Quadratic kernel '''
    def __init__(self,alpha=1):
        self.alpha =l

        self.to_precompute = set(['distance'])
    
    def __call__(self,x,y,**kwargs):
        if 'distance' not in kwargs :
            dist = cdist(x,y)
        else : 
            dist = kwargs['distance']
        return (1+ dist**2/(2*self.alpha * self.l**2) ) **(-self.alpha)