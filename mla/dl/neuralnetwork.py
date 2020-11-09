import numpy as np
from .optimizer.optimizer import GradientDescent
from .layers.inputlayer import InputLayer
from .layers.loss import Loss

class NeuralNetwork:
    def __init__(self,input_shape,loss=Loss):
        assert isinstance(loss,Loss), "loss must be an instance of Loss"
        self.layers = [InputLayer(input_shape)]
        self.loss = loss
    
    @property
    def output_layer(self):
        return self.layers[-1]

    def add(self,layer):
        layer.plug(self.output_layer)
        self.layers.append(layer)
        self.loss.plug(self.output_layer)

    def forward(self,X):
        self.loss.forward(X)

    def backprop(self,X,y):
        delta = self.loss.backprop(X,y)
        for i in range(len(self.layers)-1,0,-1):
            delta = self.layers[i].backprop(X,delta)

    
    def update(self,lr):
        for layer in self.layers[1:]:
            layer.update(lr)

    def fit(self,X,y,optimizer=GradientDescent()):
        optimizer.minimize(self,X,y)
    
    def predict(self,X):
        return self.output_layer.forward(X)

    def score(self,X,y):
        y_hat = self.predict(X)
        return self.loss.loss_function(y,y_hat)

        



