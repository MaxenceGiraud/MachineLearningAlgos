import numpy as np
from .kmedoids import Kmedoids
from ..kernels import RBF

class KernelKmedoids(Kmedoids):
    def __init__(self,k=3,kernel=RBF(),max_iter=100):
        self.kernel = kernel
        super().__init__(k=k,max_iter=max_iter)

    def _compute_dist(self,x,y):
        if hasattr(self.kernel, 'to_precompute') and ('distance' in self.kernel.to_precompute or 'distance_manhattan' in self.kernel.to_precompute):
            ## Prevent useless computations  when k(x,x) gives 1
            kxx = np.array(1)
            kyy = np.array(1)
        else :
            kxx = np.diag(self.kernel(x,x))
            kyy = np.diag(self.kernel(y,y))
        K =  kxx.reshape(-1,1) -2 * self.kernel(x,y) + kyy.reshape(1,-1)
        return K