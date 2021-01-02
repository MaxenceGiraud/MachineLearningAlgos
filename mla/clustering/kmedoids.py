import numpy as np
from scipy.spatial.distance import cdist
from .base_unsupervized import BaseUnsupervized

class Kmedoids(BaseUnsupervized):
    '''K-Medoids clustering algorithm
    Parameters
    ----------
    k : int,
        Number of centers
    max_iter : int,
        maximum number of iterations of the algo
      
    '''
    def __init__(self,k=3,max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = False
    
    def _compute_dist(self,x,y):
        return cdist(x,y)

    def fit(self,X):
        dist_matrix = self._compute_dist(X,X) 

        ## Init (kmeans++)
        centroids = [np.random.randint(0,X.shape[0])]
        for i in range(self.k-1):
            prob_dist_squared = np.min(dist_matrix[centroids],axis=0)**2
            prob_dist_squared = prob_dist_squared / sum(prob_dist_squared) # Normalize probability
            centroids.append(int(np.random.choice(np.arange(X.shape[0]),1,p=prob_dist_squared)))

        centroids = np.array(centroids)
        old_centroids = np.zeros(centroids.shape)
        i = 0
        while i < self.max_iter and np.any(old_centroids != centroids) :
            old_centroids = np.copy(centroids)
                 
            clusters = np.argmin(dist_matrix[centroids],axis=0) # Assign each point a cluster (the closest mean to the point)
            
            # Update the means with the mean of the point in the cluster
            for j in range(self.k):
                cluster_point = np.where(clusters == j)[0]
                inner_centroid_dist = np.sum(dist_matrix[cluster_point,cluster_point],axis=0) 

                centroids[j] = cluster_point[np.argmin(inner_centroid_dist)] 
            i+=1
        
        self.centroids = X[centroids]
    
    def predict(self,X):
        dist_centroids = self._compute_dist(self.centroids,X) # compute all dists between the means and every point
        return np.argmin(dist_centroids,axis=0)