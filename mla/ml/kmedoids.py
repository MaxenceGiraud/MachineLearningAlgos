import numpy as np
from scipy.spatial.distance import cdist

class Kmedoids:
    '''K-Medoids clustering algorithm'''
    def __init__(self,k,max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = False

    def fit(self,X):
        ## Init
        centroids = [X[np.random.randint(0,X.shape[0])]]
        for i in range(self.k-1):
            prob_dist_squared = np.min(cdist(centroids,X),axis=0)**2
            prob_dist_squared = prob_dist_squared / sum(prob_dist_squared) # Normalize probability
            centroids = np.vstack([centroids,X[np.random.choice(X.shape[0],1,p=prob_dist_squared)]]) 

        old_centroids = np.zeros(centroids.shape)
        i = 0
        while i < self.max_iter and np.any(old_centroids != centroids) :
            old_centroids = np.copy(centroids)
                 
            dist_centroids = cdist(centroids, X) # compute all dists between the means and every point
            clusters = np.argmin(dist_centroids,axis=0) # Assign each point a cluster (the closest mean to the point)
            
            # Update the means with the mean of the point in the cluster
            for j in range(self.k):
                cluster_point = X[clusters == j]
                inner_centroid_dist = np.sum([cdist([c],cluster_point) for c in cluster_point],axis=0)
                centroids[j] = cluster_point[np.argmin(inner_centroid_dist)] 
            i+=1
        
        self.centroids = centroids
        return self.centroids
    
    def predict(self,X):
        dist_centroids = cdist(self.centroids, X) # compute all dists between the means and every point
        return np.argmin(dist_centroids,axis=0)
        


