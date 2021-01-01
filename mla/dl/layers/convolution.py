import numpy as np
from .base_layer import BaseLayer
from ..activation import Linear
from functools import partial 
from scipy.signal import convolve,convolve2d
#scipy.ndimage.convolve with axis ??? 

# https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
# https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
# https://medium.com/@shashikachamod4u/calculate-output-size-and-number-of-trainable-parameters-in-a-convolution-layer-1d64cae6c009
# https://towardsdatascience.com/convolutional-neural-networks-f62dd896a856
# https://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

def convolution1d(M,kernel,stride,padding):
    # TODO alternative to np convolve and to add specific stride and padding values
    return

def convolution2d(M,kernel,stride,padding):
    # TODO alternative to np convolve and to add specific stride and padding values
    return

class BaseConvolutionLayer(BaseLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=True,stride=1):
        # TODO encompass stride and padding
        self.units = units
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding 
        self.padding_size = 0
        if padding != 0  and padding != False: 
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


    @property
    def nparams(self):
        return self.kernel.size + self.b.size
    
    def forward(self,X):
        self.zin = self.input_unit.forward(X) 
        for i in range(self.units):
            self.zout[i] = self.activation.f(self.convolve(self.zin,self.kernel[i])) 
        return self.zout
    
    def get_gradients(self):
        return self.dk,self.db

    def update_weights(self,weights_diff):
        uk,ub = weights_diff
        self.kernel += uk
        self.b += ub

class Conv1D(BaseConvolutionLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=True):

        assert isinstance(kernel_size,int) or len(kernel_size) ==1 , "Kernel size must be an int or array of size 1"

        if self.padding_size == 0 :
            self.convolve = partial(convolve,{'mode':"valid"})
        else : 
            raise NotImplementedError

        super().__init__(units=units,kernel_size=kernel_size,activation=activation,padding=padding)

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Conv1D layer must be a vector"

        super().plug(intputlayer)

        if self.padding :
            self.output_shape = (self.units,*self.input_shape)
        else : 
            self.output_shape  = self.input_shape + self.kernel_size -1 
            raise NotImplementedError
    
    def backprop(self,delta):
        #dk = delta.T @ self.zin 
        db = np.sum(delta,axis=0)
        # TODO

class Conv2D(BaseConvolutionLayer):
    def __init__(self,units,kernel_size,activation=Linear(),padding=True):

        assert len(kernel_size) == 2, "Kernel size must be a 2D Matrix"

        if self.padding_size == 0 :
            self.convolve = partial(convolve2d,{'mode':"valid"})
        else : 
            raise NotImplementedError

        super().__init__(units=units,kernel_size=kernel_size,activation=activation,padding=padding)

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 2, "Input of Conv2D layer must be a 2D Matrix"

        super().plug(intputlayer)

        if self.padding :
            self.output_shape = (self.units,*self.input_shape)
        else : 
            self.output_shape  = self.input_shape + self.kernel_size -1 
            raise NotImplementedError
    
    def backprop(self,delta):
        db = np.sum(delta,axis=0)
        # TODO