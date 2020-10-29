import numpy as np
from scipy.spatial.distance import cdist
from itertools import compress  

class DBSCAN:
    def __init__(self,eps=0.2,min_pts=8,metric='minkowski'):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric

    def predict(self,X):
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