import numpy as np
from scipy.spatial.distance import cdist
from ..base import BaseClassifier,BaseRegressor
from .knn import KNNRegressor,KNNClassifier
from ..kernels import RBF

class KernelKNNClassifier(KNNClassifier):
    '''K Nearest Neighbhor Classifier
    Parameters
    ----------
    k : int,
        Number of neighbors to consider
    kernel : callable,
        Kernel used to compute distance between samples
    '''
    def __init__(self,k,kernel=RBF()):
        self.k = k
        self.kernel = kernel

    def predict(self,X_test):
        if hasattr(self.kernel, 'to_precompute') and ('distance' in self.kernel.to_precompute or 'distance_manhattan' in self.kernel.to_precompute):
            ## Prevent useless computations  when k(x,x) gives 1
            kxx = 1
            kxtxt = 1
        else :
            kxx = np.diag(self.kernel(self.X,self.X))
            kxtxt = np.diag(self.kernel(X_test,X_test))
        K =  kxx.reshape(1,-1) -2 * self.kernel(X_test,self.X) + kxtxt.reshape(-1,1)
        return super().predict(X_test,dist = abs(K))

class KernelKNNRegressor(KNNRegressor):
    '''KNN Regressor with uniform weight
    Parameters
    ----------kernel
    k : int,
        Number of neighbors to consider
    kernel : string,
        Kernel used to compute distance between samples
    '''
    def __init__(self,k,kernel=RBF()):
        self.k = k
        self.kernel = kernel

    def predict(self,X_test):
        if hasattr(self.kernel, 'to_precompute') and ('distance' in self.kernel.to_precompute or 'distance_manhattan' in self.kernel.to_precompute):
            ## Prevent useless computations  when k(x,x) gives 1
            kxx = 1
            kxtxt = 1
        else :
            kxx = np.diag(self.kernel(self.X,self.X))
            kxtxt = np.diag(self.kernel(X_test,X_test))
        K =  kxx.reshape(1,-1) -2 * self.kernel(X_test,self.X) + kxtxt.reshape(-1,1)
        return super().predict(X_test,dist = abs(K))