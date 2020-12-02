import numpy as np
from .neuralnetwork import NeuralNetwork
from .layers.loss import MAE
from .layers.inputlayer import InputLayer
from .layers.dense import Dense
from copy import deepcopy,copy

class Encoded:
    '''Layer at the middle of an Autoencoder'''
    def __init__(self):
        self.output_unit = None
        self.input_unit = None
        self.mode = "linear"
    
    def set_mode(self,mode):
        self.mode = mode
    
    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape
        self.output_shape = self.input_shape
        
        self.input_unit = intputlayer

    def forward(self,X):
        if self.mode in ("linear","encoding") :
            return self.input_unit.forward(X) 
        elif self.mode == "decoding" :
            return X
        else :
            raise ValueError("mode of Encoded layer should be linear,encoding or decoding")

    def backprop(self,delta):
        return delta

    def update(self,lr):
        pass


class AutoEncoder(NeuralNetwork):
    ''' Autoencoder neural network,
     adding layer to the NN adds it automatically to the encoder and the decoder
     '''
    def __init__(self,input_shape,loss=MAE()):
        self.loss = loss
        self.input_shape = input_shape
        self.clear()
    
    def clear(self):
        self.encoder = [InputLayer(self.input_shape)]
        self.decoder = []
        self.encoded_layer = Encoded()
        self.decoder.append(Dense(np.array(sum(self.input_shape))))

        self.encoded_layer.plug(self.encoder[-1])
        self.decoder[-1].plug(self.encoded_layer)

    @property
    def output_layer(self):
        return self.decoder[0]
        
    def add(self,layer,encoding_layer=False):
        layer_decoder = deepcopy(layer)

        # Connect encoder
        layer.plug(self.encoder[-1])
        self.encoder.append(layer)
        self.encoded_layer.plug(self.encoder[-1])

        if not encoding_layer :
            # Connect decoder
            layer_decoder.plug(self.encoded_layer)
            for i in range(len(self.decoder)-2,0,-1):
                self.decoder[i].plug(self.decoder[i+1])

            self.decoder.append(layer_decoder)
        else :
            self.decoder[-1].plug(self.encoded_layer)
            for i in range(len(self.decoder)-2,0,-1):
                self.decoder[i].plug(self.decoder[i+1])

        self.loss.plug(self.output_layer)

    
    def forward(self,X):
        self.loss.forward(X,X)
    
    def backprop(self,X):
        delta = self.loss.backprop(X)
        delta_loss = np.copy(delta)

        for i in range(0,len(self.decoder)):
            delta = self.decoder[i].backprop(delta)
        for i in range(len(self.encoder)-1,0,-1):
            delta = self.encoder[i].backprop(delta)

        return delta_loss

    def update(self,lr):
        for layer in self.encoder[1:]:
            layer.update(lr)
        for layer in self.decoder:
            layer.update(lr)
    
    def score(self,X):
        X_hat = self.predict(X)
        return self.loss.loss_function(X,X_hat)

    def encode(self,X):
        return self.encoded_layer.forward(X)
    
    def decode(self,X):
        self.encoded_layer.set_mode("decoding")
        decoded =  self.decoder[0].forward(X)
        self.encoded_layer.set_mode("linear")

        return decoded
    
    def get_list_layers_todisplay(self):
        display_list = copy(self.encoder)
        d = copy(self.decoder)
        d.reverse()
        display_list.extend(d)
        return display_list