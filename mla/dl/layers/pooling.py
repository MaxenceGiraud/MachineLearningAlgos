import numpy as np
from .base_layer import BaseLayer

class BasePooling1d:
    def __init__(self,pool_size=3,padding=0,stride=1):
        assert isinstance(pool_size,int), "Pool size must be an int"
        assert isinstance(padding,int) and padding >= 0, "Padding must an int >= 0"
        self.pool_size = pool_size
        self.padding = padding
        if stride > 1 :
            raise NotImplementedError("Stride Greater than 1 is not implemented")
        self.stride = stride
    
    def _add_padding(self,X):
        '''Add padding to X'''
        X = np.hstack((np.zeros(X.shape[0]*self.padding).reshape(-1,self.padding),X)) # Left padding
        X = np.hstack((X,np.zeros(X.shape[0]).reshape(-1,self.padding)))# Right padding
        return X
        
    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Maxpooling 1d take only vector as input" 
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = [1+ self.input_shape + self.padding  - self.pool_size]
        assert self.output_shape[0] > 0 , "Shapes of pooling and input don't match"
        self.input_unit = intputlayer
        intputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X,*args,**kwargs):
        self.zin = self.input_unit.forward(X) 

        self.zout = np.apply_along_axis(self._pool,1,self._add_padding(X))
        return self.zout
    
    def _pool(self,*args,**kwargs):
        raise NotImplementedError

class MaxPooling1d(BasePooling1d,BaseLayer):

    def _pool(self,X):
        return [np.max(X[i:i+self.pool_size]) for i in range(X.shape[0]-self.pool_size+1)] 
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class MinPooling1d(BasePooling1d,BaseLayer):
    
    def _pool(self,X):
        return [np.min(X[i:i+self.pool_size]) for i in range(X.shape[0]-self.pool_size+1)] 
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class AvgPooling1d(BasePooling1d,BaseLayer):
    
    def _pool(self,X):
        return [np.mean(X[i:i+self.pool_size]) for i in range(X.shape[0]-self.pool_size+1)] 
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta