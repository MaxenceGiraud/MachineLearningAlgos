import numpy as np
from .base_layer import BaseLayer
from ..activation import Linear
from functools import partial 
from scipy.signal import convolve,convolve2d
from abc import abstractmethod


def convolution1d(M,kernel,stride,padding):
    # TODO alternative to np convolve and to add specific stride values
    raise NotImplementedError

def convolution2d(M,kernel,stride,padding):
    # TODO alternative to np convolve and to add specific stride values
    raise NotImplementedError

class BaseConvolutionLayer(BaseLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=False,stride=1):
        # TODO encompass stride and padding
        self.units = units
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding 
        self.padding_size = 0
        if  padding == True: 
            raise NotImplementedError("Padding is not yet implemented")
        self.stride = stride
        if self.stride != 1 : 
            raise NotImplementedError("Stride different from 1 is not implemented")

    def plug(self,intputlayer):
        
        self.input_shape = intputlayer.output_shape
        self.input_unit = intputlayer
        intputlayer.output_unit = self

        self.zin = 0
        self.zout = 0

        # Weights
        self.kernel = np.random.randn(self.kernel_size*self.units).reshape((self.units,*self.kernel_size))
        self.b = np.random.randn(self.units)
        # deriv
        self.dk = np.zeros(self.kernel.shape)
        self.db = np.zeros(self.b.shape)

    @abstractmethod
    def convolve(X,k):
        raise NotImplementedError

    @property
    def nparams(self):
        return self.kernel.size + self.b.size
    
    def forward(self,X):
        self.zin = self.input_unit.forward(X) 
        self.zout = np.zeros((X.shape[0],*self.output_shape))
        for i in range(X.shape[0]): #TODO vectorize with np.apply_along_axis
            for j in range(self.units):
                self.zout[i,j] = self.activation.f(self.convolve(self.zin,self.kernel[j])) 
        return self.zout
    
    def get_gradients(self):
        return self.dk,self.db

    def update_weights(self,weights_diff):
        uk,ub = weights_diff
        self.kernel += uk
        self.b += ub
    
    def backprop(self,delta):
        self.dk = np.zeros(self.kernel_size)
        for i in range(delta.shape[0]):
            self.dk = self.dk + self.convolve(np.rot90(np.rot90(delta.T)) ,self.zin)
        self.db = np.sum(delta,axis=0)
        delta = np.zeros(self.zin.shape)
        for i in range(delta.shape[0]):
            delta[i] = self.convolve(delta,self.k)
        delta = delta + self.activation.deriv(self.zin)

        return delta

class Conv1D(BaseConvolutionLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=False):

        assert isinstance(kernel_size,int) or len(kernel_size) ==1 , "Kernel size must be an int or array of size 1"

        super().__init__(units=units,kernel_size=kernel_size,x=activation,padding=padding)
    
    def convolve(X,k):
        return convolve(X,k,mode='valid')

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Conv1D layer must be a vector"
input_shape
        super().plug(intputlayer)

        if self.padding :
            self.output_shape = (self.units,*self.)
        else : 
            self.output_shape  = self.input_shape + self.kernel_size -1 
            raise NotImplementedError

class Conv2D(BaseConvolutionLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=False):

        assert len(kernel_size) == 2, "Kernel size must be a 2D Matrix"

        super().__init__(units=units,kernel_size=kernel_size,activation=activation,padding=padding)
    
    def convolve(X,k):
        return convolve2d(X,k,mode='valid')

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 2, "Input of Conv2D layer must be a 2D Matrix"

        super().plug(intputlayer)

        if self.padding :
            self.output_shape = (self.units,*self.input_shape)
        else : 
            self.output_shape  = self.input_shape + self.kernel_size -1 
            raise NotImplementedError