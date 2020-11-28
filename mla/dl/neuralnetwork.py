import numpy as np
from .optimizer.gradientdescent import GradientDescent
from .layers.inputlayer import InputLayer
from .layers.loss import Loss,MSE,MAE

class NeuralNetwork:
    '''Neural Network  '''
    def __init__(self,input_shape,loss=MAE()):
        self.layers = [InputLayer(input_shape)]
        self.loss = loss
    
    @property
    def output_layer(self):
        return self.layers[-1]

    def add(self,layer):
        layer.plug(self.output_layer)
        self.layers.append(layer)
        self.loss.plug(self.output_layer)

    def forward(self,X,y):
        return self.loss.forward(X,y)

    def backprop(self,y):
        delta = self.loss.backprop(y)
        delta_loss = np.copy(delta)
        for i in range(len(self.layers)-1,0,-1):
            delta = self.layers[i].backprop(delta)
        
        return delta_loss

    
    def update(self,lr,noise_std=0):
        for layer in self.layers[1:]:
            layer.update(lr,noise_std)

    def fit(self,X,y,optimizer=GradientDescent()):
        optimizer.minimize(self,X,y)
    
    def predict(self,X):
        return self.output_layer.forward(X)

    def score(self,X,y):
        y_hat = self.predict(X)
        return self.loss.loss_function(y,y_hat)