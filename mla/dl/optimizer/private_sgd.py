import numpy as np
from .baseoptimiser import BaseOptimizer

class PrivateSGD(BaseOptimizer):
    ''' Epsilon Delta Private Stochastic Gradient Descent'''
    def __init__(self,learning_rate=0.1,epsilon=0.1,delta=1,n_iter=100,stopping_criterion=1e-3):
        self.epsilon = epsilon
        self.delta = delta
        super().__init__(learning_rate=learning_rate,batch_size=1,n_iter=n_iter,stopping_criterion=stopping_criterion)
    
    def update(self,nn,X,*args,**kwargs):
        noise_std = 16 * np.sqrt(self.n_iter_max * np.log(2/self.delta) * np.log(max(1,1.25*self.n_iter_max/(self.delta*X.shape[0])))) / (X.shape[0]*self.epsilon)
        #print(noise_std)

        for layer in nn.get_layers_to_update():
            grads = layer.get_gradients()
            weights_updates = []
            for g in grads :
                weights_updates.append(-self.lr * (g+np.reshape(np.random.normal(0,noise_std,size=g.size),g.shape)))
            layer.update_weights(weights_updates)