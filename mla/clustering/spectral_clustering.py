import numpy as np
from .base_unsupervized import BaseUnsupervized
from ..kernels import RBF
from .kmeans import Kmeans
from scipy.spatial.distance import cdist

class SpectralClustering(BaseUnsupervized):
    ''' Spectral Clustering Algorithm 

    Parameters
    ----------
    k : int,
        Number of Clusters
    laplacian_method : str,
        ...
    adjacent_method  : str,
        KNN or Kernel
    kernel : callable,
        use to compute affinity matrix if adjacent_method== "kernel", compatible with kernels implemented in mla.kernels
    precomputed : bool,
        If true, X is considered to be the affinity matrix
    n_neighbors : int,
        Number of Neighbors to use if adjacent_method== 'KNN
    '''
    def __init__(self,k,laplacian_method='symmetric',adjacent_method='KNN',kernel=cdist,predict_method=Kmeans,precomputed=False,n_neighbors = 10):
        self.k = k 
        self.laplacian_method = laplacian_method
        self.adjacent_method = adjacent_method # KNN or None
        self.kernel = kernel  
        self.predict_method = predict_method  
        self.precomputed = precomputed
        self.n_neighbors = n_neighbors

    
    def _compute_laplacian(self,W):
        ''' Compute the Laplacian of the graph

        Parameters 
        ----------
        D : array of size (n,n),
            Adjacent matrix
        '''
        D = np.diag(np.sum(W,axis=1))  # Degree Matrix

        if self.laplacian_method == 'unnormalized':
            L = D - W

        elif self.laplacian_method == 'random_walk':
            L = np.eye(W.shape[0]) - np.linalg.pinv(D) @ W

        elif self.laplacian_method == 'symmetric':
            D_power = np.sqrt(np.linalg.pinv(D))
            L = np.eye(W.shape[0]) - D_power @ W @ D_power
        
        else :
            raise ValueError
        
        L = np.nan_to_num(L, copy=False, nan=0) # Remove nans
        return L
   
    def _compute_affinity_nearest_neighbors(self,X):
        dist = cdist(X,X)

        nearest_neighbors = np.argsort(dist,axis=1)[:,:self.n_neighbors] ## Sorting and keeping index of K nearest neighbours for each test point
        A = np.zeros(dist.shape)
        for i in range(X.shape[0]):
            A[i,nearest_neighbors[i]] = 1

        return A
        
    
    def fit_predict(self,X):
        # Affinity/Similarity matrix
        if self.precomputed :
            A = X
        elif self.adjacent_method == 'KNN' :
            A = self._compute_affinity_nearest_neighbors(X)
        elif self.adjacent_method == 'kernel' :
            A= self.kernel(X,X) 
        else :
            raise ValueError
        
        Laplacian = self._compute_laplacian(A) # Laplacian matrix

        eig_val,eig_vec = np.linalg.eig(Laplacian) 
        Xnew = eig_vec[:,np.argsort(eig_val)[:self.k]].real # Take eigenvec corresponding to k smallest eigenvalues

        y_hat = self.predict_method(self.k).fit_predict(Xnew)

        return y_hat