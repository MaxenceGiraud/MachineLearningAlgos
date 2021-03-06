import numpy as np
from .base_layer import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        pass

    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        self.output_shape = [np.prod(self.input_shape)]

        self.input_unit = inputlayer
        inputlayer.output_unit = self

    def forward(self,X):
        return X.reshape((X.shape[0],-1))

    def backprop(self,X,delta):
        return delta.reshape(self.input_shape)
    