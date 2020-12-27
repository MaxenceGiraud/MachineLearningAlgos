import numpy as np
from ..layers.dropout import Dropout


class BaseOptimizer:
    def __init__(self,learning_rate=0.1,batch_size = 32,n_iter=100,stopping_criterion=1e-3):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iter_max = n_iter
        self.stopping_criterion = stopping_criterion
    
    def update(self,nn,t,*args,**kwargs):
        pass

    def init_layers(self,nn):
        for layer in nn.get_layers_to_update():
            if isinstance(layer,Dropout):
                layer.training = True
    
    def clear_layer_training(self,nn):
        for layer in nn.get_layers_to_update():
            if isinstance(layer,Dropout):
                layer.training = False

    def minimize(self,nn,X,y,weights=1):
        # No weights 
        if weights == 1 or weights is None :
            weights = np.ones(X.shape[0])
        # add column of 1 for the bias/intercept
        self.init_layers(nn)
        g = 1
        nb_iter = 0
        batch_iter = len(y) / self.batch_size # number of mini batch
        while np.linalg.norm(g) > self.stopping_criterion and nb_iter < self.n_iter_max:
            
            # Random permutation
            permut = np.random.permutation(X.shape[0])
            X = X[permut]
            y = y[permut]
            loss = 0
            for i in range(int(batch_iter)):
                range_start, range_end = i*self.batch_size, (i+1)*self.batch_size

                loss += nn.forward(X=X[range_start:range_end],y=y[range_start:range_end],weights=weights[range_start:range_end]) # forward pass
                g += nn.backprop(y=y[range_start:range_end],weights=weights[range_start:range_end]) # backprop
                self.update(nn,t=nb_iter,X=X)  # Update weights
            # last mini batch
            nn.forward(X=X[int(batch_iter)*self.batch_size:],y=y[int(batch_iter)*self.batch_size:],weights=weights[int(batch_iter)*self.batch_size:]) # forward pass
            nn.backprop(y=y[int(batch_iter)*self.batch_size:],weights=weights[int(batch_iter)*self.batch_size:]) # backprop
            self.update(nn,t=nb_iter,X=X)  # Update weights

            print('-----\n Iter ',nb_iter)
            print("loss =",np.mean(loss)/np.ceil(batch_iter))
            print("g =",np.mean(g)/np.ceil(batch_iter))
            nb_iter += 1
            
        self.clear_layer_training(nn)