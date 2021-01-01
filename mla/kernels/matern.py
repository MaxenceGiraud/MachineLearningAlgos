import numpy as np
from scipy.special import gamma,kv
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist

class Matern(BaseKernel):
    ''' Matern Kernel
    nu : float,
        Smoothness of the kernel, as nu grows to infinity it converges to a RBF kernel
    l : float,
        lenght scale parameter
    '''
    def __init__(self,nu=1.5,l=1):
        self.nu = nu
        self.l = l

        self.to_precompute = set(['distance'])
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        if 'distance' not in kwargs :
            dist = cdist(x,y)
        else : 
            dist = kwargs['distance']
        
        x = (np.sqrt(2*self.nu)/self.l * dist)
        out = 1/(gamma(self.nu)*2**(self.nu-1)) * x**self.nu * kv(self.nu,x)
        return np.where(np.isnan(out),1,out)