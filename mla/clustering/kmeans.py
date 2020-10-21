import numpy as np
from scipy.spatial.distance import cdist

class Kmeans:
    '''K-Means clustering algorithm'''
    def __init__(self,k,max_iter=100,init='kmeans++'):
        self.k = k
        self.max_iter = max_iter
        self.cluster_means = False
        self.init = init

    def fit(self,X):
        ## Initialize the K means
        if self.init == 'random' :
            means  = X[np.random.randint(0,X.shape[0],size=self.k)] # Choose K random points in X as init
        elif self.init == "kmeans++":
            means = [X[np.random.randint(0,X.shape[0])]]
            for i in range(self.k-1):
                prob_dist_squared = np.min(cdist(means,X),axis=0)**2
                prob_dist_squared = prob_dist_squared / sum(prob_dist_squared) # Normalize probability
                means = np.vstack([means,X[np.random.choice(X.shape[0],1,p=prob_dist_squared)]]) 

        old_means = np.zeros(means.shape)
        i = 0
        while i < self.max_iter and np.any(old_means != means) :
            dist_means = cdist(means, X) # compute all dists between the means and every point
            clusters = np.argmin(dist_means,axis=0) # Assign each point a cluster (the closest mean to the point)

            old_means = np.copy(means)
            # Update the means with the mean of the point in the cluster
            for j in range(self.k):
                cluster_point = X[clusters == j]
                means[j] = np.mean(cluster_point,axis=0)
            i+=1
        
        self.cluster_means = means
        return self.cluster_means
    
    def predict(self,X):
        dist_means = cdist(self.cluster_means, X) # compute all dists between the means and every point
        return np.argmin(dist_means,axis=0)
        


