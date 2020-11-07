import numpy as np
from .optimizer.optimizer import GradientDescent
from .layers.inputunit import InputUnit
from .loss.loss import Loss

class NeuralNetwork:
    def __init__(self,input_shape):
        self.layers = [InputUnit(input_shape)]

    def add (self,layer):
        pass

    def forward(self,range):
        pass

    def backprop(self,range):
        pass

    def fit(self,X,y,optimizer=GradientDescent()):
        assert isinstance(self.layers[-1],Loss), "Last layer must be a loss"
        optimizer.minimize(self,X,y)
        



