import numpy as np
from .base_kernel import BaseKernel
from scipy.spatial.distance import cdist
from .metrics import intersection_measure

class IntersectionKernel(BaseKernel):
    ''' Intersection Kernel
    Takes as input catergorical data, and compute normalized intersection betweent them
    '''
    def __init__(self):
        pass
    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        inter =  cdist(x,y,metric=intersection_measure)  

        return inter