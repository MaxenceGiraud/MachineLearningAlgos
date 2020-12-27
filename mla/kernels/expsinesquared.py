import numpy as np
from scipy.spatial.distance import cdist
from .base_kernel import BaseKernel

class ExpSineSquared(BaseKernel):
    '''Rational Quadratic kernel 
    l :float,
        length scale parameter
    p : float,
        periodicity
    '''
    def __init__(self,l=1,p=1):
        self.p =p
        self.l = l

        self.to_precompute = set(['distance'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'distance' not in kwargs :
            dist = cdist(x,y)
        else : 
            dist = kwargs['distance']
        return np.exp(-2*np.sin(np.pi*dist/self.p)**2/self.l**2)