import numpy as np
from .baseoptimiser import BaseOptimizer 

class NAdam(BaseOptimizer):
    def __init__(self,learning_rate=0.1,batch_size = 32,momentum=lambda t,b : b*(1-0.5*0.96**(t/250)),beta1=0.9,beta2=0.99,n_iter=100,stopping_criterion=1e-3):
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        super().__init__(learning_rate=learning_rate,batch_size=batch_size,n_iter=n_iter,stopping_criterion=stopping_criterion)

    def init_layers(self,nn):
        for layer in nn.get_layers_to_update():
            layer.m = []
            layer.v = []
            for g in layer.get_gradients():
                layer.m.append(np.zeros(g.shape))
                layer.v.append(np.zeros(g.shape))
        super().init_layers(nn)
    
    def update(self,nn,t,*args,**kwargs):
        ''' Update weights of the nn
        t : int,
            N iteration
        '''
        for layer in nn.get_layers_to_update():
            grads = layer.get_gradients()
            weights_updates = []
            for i in range(len(grads)) :
                g_hat = grads[i] / (1-np.prod(self.momentum(np.arange(t+1),self.beta1))) 
                layer.m[i] = self.beta1 * layer.m[i] + (1-self.beta1)*grads[i]
                layer.v[i]  = self.beta2 * layer.v[i] + (1-self.beta2)*grads[i]**2
                m_hat = layer.m[i] / (1 - self.beta1**(t+1))
                v_hat = layer.v[i] / (1 - self.beta2**(t+1))
                m_tilda = (1-self.momentum(t,self.beta1))*g_hat + self.momentum(t+1,self.beta1) *m_hat
                weights_updates.append(-self.lr * m_tilda / (np.sqrt(v_hat) + 1e-6))
            layer.update_weights(weights_updates)