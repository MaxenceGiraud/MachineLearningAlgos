import numpy as np
from scipy.spatial.distance import cdist
from itertools import compress  


class PriorityQueue:
    def __init__(self):
        self.queue = {}

    @property
    def highest_priority(self):
        return max(self.queue.keys())
    
    @property
    def is_empty(self):
        return not bool(self.queue)

    def get(self):
        return self.queue[self.highest_priority]

    def pop(self):
        self.queue.pop(self.highest_priority)

    def move(self,item,priority): 
        key_item = [key for key,value in self.queue.items() if value==item]
        if len(key_item) >= 1 :
            key_item = key_item[0]
            self.queue.pop(key_item)
        self.queue[priority] = item

    def insert(self,item,priority):
        self.queue[priority] = item


class OPTICS:
    def __init__(self,max_eps=30,min_pts=8,metric='minkowski'):
        self.max_eps = max_eps
        self.min_pts = min_pts
        self.metric = metric

    def predict(self,X):
    
        ## INIT
        dist_all = cdist(X,X,metric=self.metric) # dist between all points
        neighbors = np.where(dist_all < self.max_eps,1,0)  # neightbors matrix
        n_neighbors = np.sum(neighbors,axis=0) # number of neighbors per point
        set_neighbors_idx =  [list(compress(range(len(bool_arr )), bool_arr )) for bool_arr in neighbors] # Boolean to indexes

        
        visited = np.zeros(X.shape[0])

        self.reachability = -np.ones(X.shape[0]) # -1 corresponds to undefined
        self.ordered_list = []
        self.clusters = []

        def core_dist(p_idx):
            if n_neighbors[p_idx] < self.min_pts :
                return -1
            else : 
                return np.min(dist_all[p_idx])

        def update(p_idx,seeds):
            coredist = core_dist(p_idx)
            for n in set_neighbors_idx[p_idx]:
                if not visited[n] :
                    new_reach_dist = max(coredist,dist_all[p_idx,n])
                    if self.reachability[n] == -1 :
                        self.reachability[n] = new_reach_dist
                        seeds.insert(n,new_reach_dist)
                    elif new_reach_dist < self.reachability[n]:
                        self.reachability[n] = new_reach_dist
                        seeds.move(n,new_reach_dist) 
            return seeds

        for i in range(X.shape[0]) :
            if not visited[i]  : 
                visited[i] = 1
                self.ordered_list.append(i)
                if core_dist(i) != -1 :
                    seeds = update(i,PriorityQueue())
                    while not seeds.is_empty:
                        q = seeds.get()
                        visited[q] = 1
                        self.ordered_list.append(q)
                        if core_dist(q) != -1 :
                            seeds = update(q,seeds)
                        seeds.pop()
                                

        return self.ordered_list,self.reachability

    
    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc
