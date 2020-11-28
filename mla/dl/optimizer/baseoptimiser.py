class BaseOptimizer:
    def __init__(self,learning_rate=0.1,batch_size = 32,n_iter=100,stopping_criterion=1e-3):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iter_max = n_iter
        self.stopping_criterion = stopping_criterion
    
    def minimize(self,nn):
        pass