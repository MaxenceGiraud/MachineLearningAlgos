import numpy as np
from .neuralnetwork import NeuralNetwork
from .layers.loss import MSE
from .layers.inputlayer import InputLayer
from .layers.dense import Dense
from .activation import Relu,Linear

class MLP(NeuralNetwork):
    def __init__(self,input_shape,layers = [20,10,1],activation=Relu,loss=MSE(),activation_last = Linear):
        self.layers = [InputLayer(input_shape)]
        self.loss = loss
        for n in layers[:-1] :
            new_layer = Dense(n,activation=activation())
            new_layer.plug(self.layers[-1])
            self.layers.append(new_layer)
        
        # Last Layer with Possibly different activation function
        new_layer = Dense(layers[-1],activation=activation_last())
        new_layer.plug(self.layers[-1])
        self.layers.append(new_layer)

        self.loss.plug(self.layers[-1])