import numpy as np


class BaseOptimizer:
    def __init__(self,learning_rate=0.1,batch_size = 32,n_iter=100,stopping_criterion=1e-3):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iter_max = n_iter
        self.stopping_criterion = stopping_criterion
    
    def update(self,nn,t,*args,**kwargs):
        pass

    def minimize(self,nn,X,y):
        # add column of 1 for the bias/intercept
        g = 1
        nb_iter = 0
        batch_iter = len(y) / self.batch_size # number of mini batch
        while np.linalg.norm(g) > self.stopping_criterion and nb_iter < self.n_iter_max:
            print(nb_iter)
            # Random permutation
            permut = np.random.permutation(X.shape[0])
            X = X[permut]
            y = y[permut]
            loss = 0
            for i in range(int(batch_iter)):
                range_start, range_end = i*self.batch_size, (i+1)*self.batch_size

                loss += nn.forward(X[range_start:range_end],y[range_start:range_end]) # forward pass
                g += nn.backprop(y[range_start:range_end]) # backprop
                self.update(nn,t=nb_iter,X=X)  # Update weights
            # last mini batch
            nn.forward(X[int(batch_iter)*self.batch_size:],y[int(batch_iter)*self.batch_size:]) # forward pass
            nn.backprop(y[int(batch_iter)*self.batch_size:]) # backprop
            self.update(nn,t=nb_iter,X=X)  # Update weights

            #g= np.mean(g,axis=0)
            print("loss :",np.mean(loss)/np.ceil(batch_iter))
            print("g :",np.mean(g)/np.ceil(batch_iter))
            nb_iter += 1