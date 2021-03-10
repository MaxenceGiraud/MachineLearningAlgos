import numpy as np
from .base_layer import BaseLayer

class Reshape(BaseLayer):
    def __init__(self,output_shape):
        self.output_shape = output_shape

    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        
        if np.prod(self.input_shape) != np.prod(self.input_shape):
            raise Exception("Input shape and output shape of Reshape Layer do not match")

        self.input_unit = inputlayer   
        inputlayer.output_unit = self

    def forward(self,X):
        return X.reshape((-1,*self.output_shape))

    def backprop(self,X,delta):
        return delta.reshape(self.input_shape)
    