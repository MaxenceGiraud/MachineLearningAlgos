import numpy as np
from .baseoptimiser import BaseOptimizer

class GradientDescent(BaseOptimizer):
    ''' Mini Batch Gradient Descent algorithm'''
    
    def update(self,nn,*args,**kwargs):
        for layer in nn.get_layers_to_update():
            grads = layer.get_gradients()
            weights_updates = []
            for g in grads :
                weights_updates.append(-self.lr * g)
            layer.update_weights(weights_updates)
        
class StochasticGradientDescent(GradientDescent):
    ''' Stochastic Gradient Descent algorithm'''
    def __init__(self,learning_rate=0.1,n_iter=100,stopping_criterion=1e-6):
        super().__init__(learning_rate=learning_rate,batch_size=1,n_iter=n_iter,stopping_criterion=stopping_criterion)
