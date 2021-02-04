import numpy as np
from .base_layer import BaseLayer
from ..activation import Linear

def sgm(x):
    return 1/(1+np.exp(-x))

class LSTM(BaseLayer):
    def __init__(self,units,activation=Linear(),return_all=False):
        self.units = units 
        self.activation = activation
        self.return_all = return_all
    
    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        self.input_unit = inputlayer
        inputlayer.output_unit = self

        self.zin = 0
        self.zout = 0

        ### Init Weights
        # Update Gate
        self.wr = np.zeros(1) 
        self.br = np.zeros(self.units) # To init with large values

        # Reset Gate
        self.wu = 0
        self.bu = 0

        # Innov Gate
        self.winnov = 0
        self.binnov = 0

        # Init Deriv
        self.dwr = np.zeros(self.wr.shape)
        self.dbr = np.zeros(self.br.shape)
        self.dwu = np.zeros(self.wu.shape)
        self.dbu = np.zeros(self.bu.shape)
        self.dwinnov = np.zeros(self.winnov.shape)
        self.dbinnov = np.zeros(self.binnov.shape)


    @property
    def nparams(self):
        return self.wr.size + self.br.size + self.wu.size + self.bu.size + self.winnov.size + self.binnov.size

    def forward(self,X):
        # Concatenate
        s = np.concatenate((self.zout,X))

        u = sgm(self.wu @ s + self.bu)  # Update gate
        r = sgm(self.wr @ s + self.br) # Reset Gate

        sp = np.concatenate((X,r*self.zout))
        zp = np.tanh(self.winnov @ sp + self.binnov) # Innov Gate

        self.zout = (1-u)*zp + self.zout*u

        return self.zout

    def get_gradients(self):
        return self.dwr, self.dbr, self.dwu, self.dbu, self.dwinnov, self.dbinnov

    def update_weights(self,weights_diff):
        uwr,ubr,uwu,ubu,uwinnov,ubinnov = weights_diff
        self.wr += uwr
        self.br += ubr
        self.wu += uwu
        self.bu += ubu
        self.winnov += uwinnov
        self.binnov += ubinnov
    
    def backprop(self,delta):
        raise NotImplementedError
        # return delta