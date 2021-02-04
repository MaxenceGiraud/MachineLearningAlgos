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
        # Forget Gate
        self.wf = np.zeros(1) 
        self.bf = np.zeros(self.units) # To init with large values

        # Input Gate
        self.wi = 0
        self.bi = 0

        # G Gate
        self.wg = 0
        self.bg = 0

        # Output Gate
        self.wo = 0
        self.bo = 0

        # Param of previous forward pass
        self.c_old = 0

        # Init Deriv
        self.dwf = np.zeros(self.wf.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dwi = np.zeros(self.wi.shape)
        self.dbi = np.zeros(self.bi.shape)
        self.dwg = np.zeros(self.wg.shape)
        self.dbg = np.zeros(self.bg.shape)
        self.dwo = np.zeros(self.wo.shape)
        self.dbo = np.zeros(self.bo.shape)


    @property
    def nparams(self):
        return self.wf.size + self.bf.size + self.wi.size + self.bi.size + self.wg.size + self.bg.size + self.wo.size + self.bo.size

    def forward(self,X):
        # Concatenate
        u = np.concatenate((self.zout,X))

        f = sgm(self.wf @ u + self.bf)  # Forget gate
        i = sgm(self.wi @ u + self.bi) # Input Gate
        h =  np.tanh(self.wg @ u + self.bg) # G Gate
        o = sgm(self.wo @ u + self.bo) # Output Gate

        c = f*self.c_old +i 
        self.zout = np.tanh(c) * o 

        return self.zout

    def get_gradients(self):
        return self.dwf, self.dbf, self.dwi, self.dbi, self.dwg, self.dbg, self.dwo, self.dbo

    def update_weights(self,weights_diff):
        uwf,ubf,uwi,ubi,uwg,ubg,uwo,ubo = weights_diff
        self.wf += uwf
        self.bf += ubf
        self.wi += uwi
        self.bi += ubi
        self.wg += uwg
        self.bg += ubg
        self.wo += uwo
        self.bo += ubo
    
    def backprop(self,delta):
        raise NotImplementedError
        # return delta