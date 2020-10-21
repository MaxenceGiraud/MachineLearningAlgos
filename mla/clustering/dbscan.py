import numpy as np
from scipy.spatial.distance import cdist
from itertools import compress  


class DBSCAN:
    def __init__(self,eps=0.3,min_pts=8,metric='minkowski'):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric

    def predict(self,X):
        #self.labels = set(y) - {-1}
        # y= -1 not labeled
        
        y_hat = np.zeros(X.shape[0])

        dist_all = cdist(X,X,metric=self.metric=)
        neighbors = np.where(dist_all < self.eps,1,0) 
        n_neighbors = np.sum(neighbors,axis=0)
        
        # Classify points as noise/core/border point
        is_noise = np.where(n_neighbors == 0,1,0)
        is_core = np.where(n_neighbors >= self.min_pts,1,0) 
        
        # Boolean to indexes
        set_neighbors_idx =  [list(compress(range(len(bool_arr )), bool_arr )) for bool_arr in neighbors[is_core]]

        # Group the clusters TODO : TO rework
        clusters = np.array,union_find(set_neighbors_idx)

        y_hat[is_noise] = -1
        for i in range(len(clusters)):
            y_hat[clusters] = i

        return y_hat
    
    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc
        

def union_find(lis):
    lis = map(set, lis)
    unions = []
    for item in lis:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions
