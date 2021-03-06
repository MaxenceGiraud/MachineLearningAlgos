import numpy as np
from .base_layer import BaseLayer

class Dropout(BaseLayer):
    def  __init__(self,proba=0.2):
        self.proba = proba
        self.training = False # Determine if model is in training mode or not
    
    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        self.output_shape = self.input_shape

        self.input_unit = inputlayer
        inputlayer.output_unit = self

    def forward(self,X):
        self.zin = self.input_unit.forward(X)
        self.zout = np.copy(self.zin)
        if self.training :
            self.mask = np.random.binomial(n=1,p=1-self.proba,size=self.zout.shape) * 1/(1-self.proba)
            self.zout= self.zout*self.mask
        return self.zout

    def backprop(self,delta):
        return delta * self.mask