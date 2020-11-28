import numpy as np
from .baseoptimiser import BaseOptimizer

class PrivateSGD(BaseOptimizer):
    ''' Epsilon Delta Private Stochastic Gradient Descent'''
    def __init__(self,learning_rate=0.1,epsilon=0.1,delta=1,n_iter=100,stopping_criterion=1e-3):
        self.epsilon = epsilon
        self.delta = delta
        super().__init__(learning_rate,1,n_iter,stopping_criterion)

    def minimize(self,nn,X,y):
        # add column of 1 for the bias/intercept
        g = 1
        nb_iter = 0
        noise_std = 16 * np.sqrt(self.n_iter_max * np.log(2/self.delta) * np.log(1.25*self.n_iter_max/(self.delta*X.shape[0]))) / (X.shape[0]*self.epsilon)
        
        while np.linalg.norm(g) > self.stopping_criterion and nb_iter < self.n_iter_max:
            print(nb_iter)
            # Random permutation
            permut = np.random.permutation(X.shape[0])
            X = X[permut]
            y = y[permut]
            loss = 0
            for i in range(X.shape[0]):
                loss += nn.forward(X[i],y[i]) # forward pass
                g += nn.backprop(y[i]) # backprop
                nn.update(self.lr)  # Update weights
           
            nn.update(self.lr,noise_std=noise_std)  # Update weights

            print("loss :",np.mean(loss)/np.ceil(batch_iter))
            print("g :",np.mean(g)/np.ceil(batch_iter))
            nb_iter += 1
 