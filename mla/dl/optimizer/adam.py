import numpy as np
from .baseoptimiser import BaseOptimizer 

class Adam(BaseOptimizer):
    def __init__(self,learning_rate=0.1,batch_size = 32,beta1=0.9,beta2=0.99,n_iter=100,stopping_criterion=1e-3):
        self.beta1 = beta1
        self.beta2 = beta2
        super().__init__(learning_rate=learning_rate,batch_size=batch_size,n_iter=n_iter,stopping_criterion=stopping_criterion)

    def init_moments(self,nn):
        for l in nn.get_layers_to_update():
            l.m = 0
            l.v = 0
    
    def update(self,nn,t,*args,**kwargs):
        ''' Update weights of the nn
        t : int,
            N iteration
        '''
        for layer in nn.get_layers_to_update(nn):
            grads = layer.get_gradients()
            weights_updates = []
            for g in grads :
                layer.m = self.beta1 * layer.m + (1-self.beta1)*g
                layer.v  = self.beta2 * layer.v + (1-self.beta2)*g**2
                m_hat = layer.m / (1 - self.beta1**t)
                v_hat = layer.v / (1 - self.beta2**t)
                weights_updates.append(-self.lr * m_hat / (np.sqrt(v_hat) + 1e-6))
            layer.update_weights(weights_updates)
        
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
                nn.update(self.lr)  # Update weights
            # last mini batch
            nn.forward(X[int(batch_iter)*self.batch_size:],y[int(batch_iter)*self.batch_size:]) # forward pass
            nn.backprop(y[int(batch_iter)*self.batch_size:]) # backprop
            nn.update(self.lr)  # Update weights

            #g= np.mean(g,axis=0)
            print("loss :",np.mean(loss)/np.ceil(batch_iter))
            print("g :",np.mean(g)/np.ceil(batch_iter))
            nb_iter += 1