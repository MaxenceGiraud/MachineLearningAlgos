import numpy as np
from .base_unsupervized import BaseUnsupervized
from ..kernels import RBF
from .kmeans import Kmeans
from ..base_adjacent import BaseAdjacent

class SpectralClustering(BaseUnsupervized,BaseAdjacent):
    ''' Spectral Clustering Algorithm 

    Parameters
    ----------
    k : int,
        Number of Clusters
    laplacian_method : str,
        unnormalized, symmetric (Default) or random_walk
    adjacent_method  : str,
        KNN, Kernel, precomputed (then X is considered to be the adjancy matrix) or epsilon
    kernel : callable,
        use to compute affinity matrix if adjacent_method== "kernel" or compute distances if kernel == "KNN", compatible with kernels.implemented in mla.kernels.
        Default to classical Euclidean distance
    predict_method : class of clustergin algo,
        Algo used to perform the clustering after the transformation of the data
    n_neighbors : int,
        Number of Neighbors to use if adjacent_method== 'KNN'
    epsilon :float,
        Max distance to consider 2 points neighbors, considered on if  adjacent_method=='epsilon'
    '''
    def __init__(self,k,laplacian_method='symmetric',adjacent_method='KNN',kernel=RBF(),predict_method=Kmeans,n_neighbors = 10,epsilon=0.5):
        self.k = k 
        self.laplacian_method = laplacian_method
        self.adjacent_method = adjacent_method # KNN or None
        self.kernel = kernel  
        self.predict_method = predict_method  
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    
    def _compute_laplacian(self,A):
        ''' Compute the Laplacian of the graph

        Parameters 
        ----------
        A : array of size (n,n),
            Adjacent matrix
        '''
        D = np.diag(np.array(np.sum(A,axis=1)).flatten())  # Degree Matrix

        if self.laplacian_method == 'unnormalized':
            L = D - A

        elif self.laplacian_method == 'random_walk':
            L = np.eye(A.shape[0]) - np.linalg.pinv(D) @ A

        elif self.laplacian_method == 'symmetric':
            D_power = np.sqrt(np.linalg.pinv(D))
            L = np.eye(A.shape[0]) - D_power @ A @ D_power
        
        else :
            raise ValueError
        
        L = np.nan_to_num(L, copy=False, nan=0) # Remove nans
        return L
  
    def transform(self,X):

        # Affinity/Similarity/Adjacent matrix
        if self.adjacent_method == 'precomputed' :
            A = X
        elif self.adjacent_method == 'KNN' :
            A = self._compute_adjacent_nearest_neighbors(X)
        elif self.adjacent_method == 'kernel' :
            A= self.kernel(X,X) 
        elif self.adjacent_method == 'epsilon':
            A = self._compute_adjacent_epsilon_neighborhood(X)
        else :
            raise ValueError
        
        Laplacian = self._compute_laplacian(A) # Laplacian matrix

        eig_val,eig_vec = np.linalg.eig(Laplacian) 
        Xnew = eig_vec[:,np.argsort(eig_val)[:self.k]].real # Take eigenvec corresponding to k smallest eigenvalues

        return Xnew
    
    def fit_predict(self,X):
        
        Xnew = self.transform(X)

        y_hat = self.predict_method(self.k).fit_predict(Xnew)

        return y_hat