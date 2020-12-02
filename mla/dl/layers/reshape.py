import numpy as np
from .base_layer import BaseLayer

class Reshape(BaseLayer):
    def __init__(self,output_shape):
        self.output_shape = output_shape

    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape
        
        if np.cumsum(self.input_shape) != np.cumsum(self.input_shape):
            raise Exception("Input shape and output shape of Reshape Layer do not match")

        self.input_unit = intputlayer   
        intputlayer.output_unit = self

    def forward(self,X):
        return X.reshape((-1,*self.output_shape))

    def backprop(self,X,delta):
        return delta.reshape(self.input_shape)
    
    def update(self,*args,**kwargs):
        pass
    