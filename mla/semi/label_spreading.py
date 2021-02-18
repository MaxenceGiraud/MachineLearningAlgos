import numpy as np
from .base_semi import BaseSemi
from ..kernels import RBF
from ..base_adjacent import BaseAdjacent

class LabelSpreading(BaseAdjacent,BaseSemi):
    ''' Label Spreading 

    Parameters
    ---------
    mu : float,
        Clamping Factor, must be between 0 and 1
    similarity : str,
        Method to compute Similiarity/Adjacency Matrix. 
    n_neighbors : int,
        Number of Neighbors to use if similarity== 'knn'
    kernel : callable,
        use to compute affinity matrix if adjacent_method== "kernel" or compute distances if kernel == "KNN", compatible with kernels.implemented in mla.kernels.
        Default to classical RBF/Gaussian Kernel
    max_iter : int,
        Maximum number of iteration
    epsilon :float,
        Max distance to consider 2 points neighbors, considered on if  similarity=='epsilon'
    '''
    def __init__(self,mu=0.2,similarity='knn',n_neighbors = 10,kernel = RBF(),max_iter=200,epsilon=0.8):
        assert similarity in ['knn','kernel','precomputed','epsilon'],"Similarity method must be one of the following : 'knn','kernel','precomputed' "
        self.mu = mu
        self.similarity = similarity
        self.kernel = kernel
        self.max_iter = max_iter

        super().__init__(n_neighbors=n_neighbors,epsilon=epsilon)

    
    def fit_predict(self,X,y):
        if self.similarity == 'knn' : 
            A = self._compute_adjacent_nearest_neighbors(X)
        elif self.similarity == 'kernel' :
            A = self.kernel(X,X)
        elif self.similarity == "epsilon" : 
            A = self._compute_adjacent_epsilon_neighborhood(X)
        elif self.similarity == 'precomputed' : 
            A = X
        else : raise ValueError

        D = np.diag(np.array(np.sum(A,axis=1)).flatten())  # Degree Matrix
        D_power = np.sqrt(np.linalg.pinv(D))
        L =  D_power @ A @ D_power # Compute Normalized Laplacian - np.eye(A.shape[0]) -

        # Init
        y_hat = np.copy(y)


        # Iterative solution
        # i = 0

        # while np.linalg.norm(y_hat-y_old) > 1e-3 and i< self.max_iter : # until convergence
        #     y_old = np.copy(y_hat)

        #     y_hat = 1/(1+self.mu) * L  @ y_hat + self.mu/(1+self.mu) * y0

        #     i += 1
        
        y_hat = self.mu/(1+self.mu) * np.linalg.pinv(np.eye(L.shape[0]) - 1/(1+self.mu) * L) @ y # Closed form solution

        return  np.sign(y_hat)