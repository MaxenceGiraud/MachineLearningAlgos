import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class BaseAdjacent:
    def __init__(self):
        self.epsilon = 0.5
        self.n_neighbors = 10

    def _compute_adjacent_nearest_neighbors(self,X):
        dist = cdist(X,X)

        nearest_neighbors = np.argsort(dist,axis=1)[:,:self.n_neighbors] ## Sorting and keeping index of K nearest neighbours for each test point
        A = csr_matrix(dist.shape)
        for i in range(X.shape[0]):
            A[i,nearest_neighbors[i]] = 1

        return A
    
    def _compute_adjacent_epsilon_neighborhood(self,X):
        dist = cdist(X,X)
        A = np.where(dist<self.epsilon,1,0)

        return A