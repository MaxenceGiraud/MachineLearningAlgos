from .base_layer import BaseLayer

class ActivationLayer(BaseLayer):
    ''' Layer that is only an activation function '''
    def __init__(self,activation):
        self.activation = activation
    
    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        self.output_shape = self.input_shape

        self.input_unit = inputlayer
        inputlayer.output_unit = self
    
    def forward(self,X,*args,**kwargs):
        return self.activation.f(X)
    
    def backprop(self,delta,*args,**kwargs):
        return self.activation.deriv(delta)