import numpy as np

class BaseOptimizer:
    def __init__(self,learning_rate=0.1,batch_size = 32,n_iter=100,eps=1e-6):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iter_max = n_iter
        self.eps = eps
    
    def minimize(self,nn):
        pass

class GradientDescent(BaseOptimizer):
    def minimize(self,nn,X,y):
        # add column of 1 for the bias/intercept
        g = 1
        nb_iter = 0
        while np.linalg.norm(g) > self.eps and nb_iter < self.n_iter_max:
            # Random permutation
            permut = np.random.permutation(X.shape[0])
            X = X[permut]
            y = y[permut]

            for i in range(len(y) // self.batch_size):
                range_start, range_end = i*self.batch_size, (i+1)*self.batch_size

                nn.forward(X[range_start,range_end]) # forward pass
                g = nn.backprop(X[range_start,range_end]) # backprop
                nn.update(self.lr)  # Update weights
            # last mini batch
            nn.forward(X[i*self.batch_size:]) # forward pass
            g = nn.backprop(X[i*self.batch_size:]) # backprop
            nn.update(self.lr)  # Update weights

            nb_iter += 1

class StochasticGradientDescent(GradientDescent):
    def __init__(self,learning_rate=0.1,n_iter=100,eps=1e-6):
        super.__init__(learning_rate=learning_rate,batch_size=1,n_iter=n_iter,eps=eps)

class Adam(BaseOptimizer):
    pass


class Adagrad(BaseOptimizer):
    pass
