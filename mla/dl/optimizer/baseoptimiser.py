class BaseOptimizer:
    def __init__(self,learning_rate=0.1,batch_size = 32,n_iter=100,eps=1e-6):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iter_max = n_iter
        self.eps = eps
    
    def minimize(self,nn):
        pass