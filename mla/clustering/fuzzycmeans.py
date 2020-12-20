import numpy as np
from scipy.spatial.distance import cdist
from .base_unsupervized import BaseUnsupervized

class FuzzyCmeans(BaseUnsupervized):
    '''Fuzzy C-means clustering algorithm
    Parameters
    ----------
    n_cluster : int,
        number of cluster
    p : float,
        fuzzyness, >0
    iter_max : int,
        Number of maximum iterations
    '''

    def __init__(self,n_cluster,p=2,iter_max=100):
        assert n_cluster>0, "n_cluster must be greater than 0"
        assert p>0, "p must be greater than 0"
        assert iter_max > 0, "iter_max must be greater than 0"

        self.n_cluster = n_cluster
        self.clusters = None
        self.p = p
        self.iter_max = iter_max
    
    def _compute_weights(self,X):
        dist = cdist(X,self.centroids)
        dist = (1/dist)** (1/(self.p-1))
        w =  dist / dist.sum(axis=1).reshape(-1,1)
        return w
    
    def fit(self,X):
        self.clusters  = np.random.random((self.n_cluster,X.shape[1]))

        w = np.random.dirichlet(np.ones(self.n_cluster),size=X.shape[0]) 
        w_old = np.random.dirichlet(np.ones(self.n_cluster),size=X.shape[0]) 
        i=0
        while i< self.iter_max and np.abs(w_old - w).sum() > 1e-6 :
            w_old = w
            
            # Update centroids
            wp = w**self.p
            self.centroids = wp.T @ X
            self.centroids /= wp.sum(axis=0).reshape(-1,1)

            # Update weights
            w = self._compute_weights(X)
            
            i+=1
       
    def predict(self,X):
        w = self._compute_weights(X)
        cluster = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cluster[i] = np.random.choice(self.n_cluster,p=w[i])
        
        return cluster