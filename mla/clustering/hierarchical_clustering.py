import numpy as np
from .base_unsupervized import BaseUnsupervized
from scipy.spatial.distance import cdist

class HierarchicalClustering(BaseUnsupervized):
    ''' (Agglomerative) Hierarchical Clustering Algorithm 

    Parameters
    ----------
    n_clusters : int,
        Number of Clusters
    linkage : str,
        Linkage criterion to use. Defaults to single
    metric : str/function,
        Metric to compute the distances. Defaults to Euclidean.
    precomputed : bool,
        If True the input matrix is considered to be the distance matrix. Defaults to False.
    distance_threshold : float,
        Minimum distances between points to continue the clustering. Defaults to 0.
    '''
    def __init__(self,n_clusters=None,linkage = "single",metric="euclidean", precomputed = False,distance_threshold = 0):
        self.n_clusters = n_clusters
        self.linkage = linkage
        assert self.linkage in ["single"] , "Linkage must be in the list : single."
        self.metric = metric
        self.precomputed = precomputed
        self.distance_threshold = distance_threshold

    
    def fit_predict(self,X):
        if self.precomputed :
            dist = X
        else :
            dist = cdist(X,X,metric=self.metric)

        i,j = np.diag_indices_from(dist)
        dist[i,j] = np.inf # Set diag to inf 

        clusters = np.arange(X.shape[0])

        while np.unique(clusters).size > self.n_clusters :

            i,j = divmod(dist.argmin(), dist.shape[1]) # Find 2 closest point

            if dist[i,j] < self.distance_threshold :
                break

            # Regroup the 2 clusters
            clusters_j_idx = np.where(clusters == clusters[j])[0]
            clusters_i_idx = np.where(clusters == clusters[i])[0]
            clusters[clusters_j_idx] = clusters[i] 

            # To not select points within the new cluster
            for i in clusters_i_idx : 
                for j in clusters_j_idx : 
                    dist[i,j] = np.inf
                    dist[j,i] = np.inf
        
        return clusters,dist