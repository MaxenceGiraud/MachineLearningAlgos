import numpy as np
from .base_semi import BaseSemi
from ..kernels import RBF
from ..base_adjacent import BaseAdjacent

class LabelPropagation(BaseAdjacent,BaseSemi):
    ''' Label Propagation 

    Parameters
    ---------
    similarity : str,
        Method to compute Similiarity/Adjacency Matrix. 
    n_neighbors : int,
        Number of Neighbors to use if similarity== 'knn'
    kernel : callable,
        use to compute affinity matrix if adjacent_method== "kernel" or compute distances if kernel == "KNN", compatible with kernels.implemented in mla.kernels.
        Default to classical RBF/Gaussian Kernel
    max_iter : int,
        Maximum number of iteration of harmonical algorithm
    epsilon :float,
        Max distance to consider 2 points neighbors, considered on if  similarity=='epsilon'
    '''
    
    def __init__(self,similarity='knn',n_neighbors = 10,kernel = RBF(),max_iter=200,epsilon=0.8):
        assert similarity in ['knn','kernel','precomputed','epsilon'],"Similarity method must be one of the following : 'knn','kernel','precomputed','epsilon' "
        self.similarity = similarity
        self.kernel = kernel
        self.max_iter = max_iter

        super().__init__(n_neighbors=n_neighbors,epsilon=epsilon)

    def _harmonic_algo(self,S,y):
        labeled = np.where(y!=0) # Labeled datapoints

        Dinv = np.linalg.pinv(np.diag(np.array(np.sum(S,axis=1)).flatten()))  # Degree Matrix

        # Init
        y_hat = np.random.randint(0,1,size=y.size)
        y_old = np.copy(y)
        i = 0

        while np.linalg.norm(y_hat-y_old) > 1e-3 and i< self.max_iter : # until convergence
            y_old = y_hat

            y_hat = Dinv @ S @ y_hat

            y_hat[labeled] = y[labeled]

            i += 1
        return np.sign(y_hat)

    
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

        y_hat = self._harmonic_algo(A,y)

        return y_hat