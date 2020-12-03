import numpy as np
from .baseoptimiser import BaseOptimizer

class Adagrad(BaseOptimizer):
  
    def update(self,nn,*args,**kwargs):
        ''' Update weights of the nn  '''
        for layer in nn.get_layers_to_update():
            grads = layer.get_gradients()
            weights_updates = []
            for g in grads :
                weights_updates.append(- self.lr / np.sqrt(np.sum(g**2) + 1e-6) * g)
            layer.update_weights(weights_updates)