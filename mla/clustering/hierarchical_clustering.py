import numpy as np
from .base_unsupervized import BaseUnsupervized
from scipy.spatial.distance import cdist
from itertools import  combinations

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

        ## Define linkage step function
        if self.linkage =="single":
            self.linkage_step = self._single_linkage
        elif self.linkage =="complete":
            self.linkage_step = self._complete_linkage
        elif self.linkage =="average":
            self.linkage_step = self._avg_linkage
        elif self.linkage =="centroid" : 
            self.linkage_step = self._centroid_linkage
        else :
            raise ValueError("Linkage must be in the list : single,complete,average,centroid.")

        self.metric = metric
        self.precomputed = precomputed
        self.distance_threshold = distance_threshold
    
    def _single_linkage(self,dist,clusters,**args):
        return divmod(dist.argmin(), dist.shape[1]) # Find 2 closest point
    

    def _full_linkage(self,dist,clusters,fun=np.max):
        clusters_unique = np.unique(clusters)
        c_idx = [np.where(clusters == clusters[ci])[0] for ci in clusters_unique]
        # Find max distance between all clusters
        mini,minj = 0,0
        min_d = np.inf
        for i,j in combinations(range(clusters_unique.size),2):
            d = fun(dist[c_idx[i]][:,c_idx[j]])
            if d < min_d : 
                min_d = d
                mini,minj = i,j

        # dist_clusters = [fun(dist[i][:,j]) for i,j in combinations(c_idx,2)]
        # ic,jc = divmod(np.argmin(dist_clusters), clusters_unique.size)

        return c_idx[mini][0],c_idx[minj][0]

    def _complete_linkage(self,dist,clusters,**args):
        return self._full_linkage(dist,clusters,np.max)

    def _avg_linkage(self,dist,clusters,**args):
        return self._full_linkage(dist,clusters,np.mean)
         
    def _centroid_linkage(self,dist,clusters,X,**args):
        clusters_unique = np.unique(clusters)

        # Find data points assigned to clusters
        c_idx = [np.where(clusters == clusters[ci])[0] for ci in clusters_unique]

        # Compute centroids
        centroids = [np.mean(X[c],axis=0) for c in c_idx]

        # Compute dist between centroids
        dist_centroids = cdist(centroids,centroids,metric=self.metric)

        i,j = np.diag_indices_from(dist_centroids)
        dist_centroids[i,j] = np.inf # Set diag to inf 

        # Find closest centroids
        ic,jc = divmod(dist_centroids.argmin(),dist_centroids.shape[1])

        return clusters_unique[ic],clusters_unique[jc]
    
    def fit_predict(self,X):
        if self.precomputed :
            dist = X
        elif self.linkage == 'centroid':
            dist = np.zeros((X.shape[0],X.shape[0]))
        else :
            dist = cdist(X,X,metric=self.metric)

        i,j = np.diag_indices_from(dist)
        dist[i,j] = np.inf # Set diag to inf 

        clusters = np.arange(X.shape[0])
        
        while np.unique(clusters).size > self.n_clusters :
            i,j = self.linkage_step(dist,clusters,X=X)

            if dist[i,j] < self.distance_threshold and self.linkage=="single":
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