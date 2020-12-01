import numpy as np
from scipy.spatial.distance import cdist
from itertools import compress  

class DBSCAN:
    ''' DBSCAN clustering algorithm 
    Ref : Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise”. In: Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Parameter
    ---------
    eps : float,
        Radius of the neighborhood of one point 
    min_pts : int,
        Minimum number of points in the neighborhood of a point to be considered a core
    metric : string,
        Metric type for the computation of distance between the points
    '''
    def __init__(self,eps=2,min_pts=8,metric='minkowski'):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric

    def fit_predict(self,X):
        #self.labels = set(y) - {-1}
        # y= -1 not labeled
        
        dist_all = cdist(X,X,metric=self.metric)
        neighbors = np.where(dist_all < self.eps,1,0) 
        n_neighbors = np.sum(neighbors,axis=0)
        
        # Classify points as noise/core/border point
        is_noise = np.where(n_neighbors == 0,1,0)
        is_core = np.where(n_neighbors >= self.min_pts,1,0) 
        
        # Boolean to indexes
        set_neighbors_idx =  [list(compress(range(len(bool_arr )), bool_arr )) for bool_arr in neighbors]

        visited = np.zeros(X.shape[0])
        clusters = []
  
        def add_neighbhor(point):
            current_cluster.append(point)
            visited[point] = 1
            for p in set_neighbors_idx[point] :
                if not visited[p] :
                    add_neighbhor(p)

        for i in range(X.shape[0]) : 
            if not visited[i]:
                visited[i] = 1
                if is_core[i] :
                    current_cluster = [i]
                    for n in set_neighbors_idx[i] :
                        add_neighbhor(n)

                    clusters.append(current_cluster)
                            
        y_hat = -np.ones(X.shape[0])
                       
        for i in range(len(clusters)):
            y_hat[clusters[i]] = i

        return y_hat
    
    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc
