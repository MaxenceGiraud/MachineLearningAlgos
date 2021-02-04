import numpy as np
from .base_layer import BaseLayer


class BasePadding1d:
    def _add_padding(self,X):
        '''Add padding to X'''
        if self.padding == 0 :
            return X
        X = np.hstack((np.zeros(X.shape[0]*self.padding).reshape(-1,self.padding),X)) # Left padding
        X = np.hstack((X,np.zeros(X.shape[0]).reshape(-1,self.padding)))# Right padding
        return X

class BasePooling1d(BasePadding1d):
    def __init__(self,pool_size=3,padding=0,stride=1):
        assert isinstance(pool_size,int), "Pool size must be an int"
        assert isinstance(padding,int) and padding >= 0, "Padding must an int >= 0"
        self.pool_size = pool_size
        self.padding = padding
        assert stride > 0, "Stride must be at min 1"
        if stride > 1 :
            raise NotImplementedError("Stride Greater than 1 is not implemented")
        self.stride = stride
        
    def plug(self,inputlayer):
        assert len(inputlayer.output_shape) == 1, "1D Pooling take only vector as input" 
        self.input_shape = inputlayer.output_shape[0]
        self.output_shape = [1+ self.input_shape + self.padding  - self.pool_size]
        assert self.output_shape[0] > 0 , "Shapes of pooling and input don't match"
        self.input_unit = inputlayer
        inputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X,*args,**kwargs):
        self.zin = self.input_unit.forward(X) 

        self.zout = np.apply_along_axis(self._pool,1,self._add_padding(self.zin))
        return self.zout
    
    def _pool(self,X,fun=np.max):
        return np.array([fun(X[i:i+self.pool_size]) for i in range(X.shape[0]-self.pool_size+1)])

class MaxPooling1d(BasePooling1d,BaseLayer):

    def _pool(self,X):
        return super()._pool(X,fun=np.max)

    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class MinPooling1d(BasePooling1d,BaseLayer):
    
    def _pool(self,X):
        return super()._pool(X,fun=np.min)
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class AvgPooling1d(BasePooling1d,BaseLayer):
    
    def _pool(self,X):
        return super()._pool(X,fun=np.mean)
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta  

class BasePadding2d:
    def _add_padding(self,X):
        '''Add padding to X'''
        if self.padding == 0 :
            return X
        def pad_1datapoint(x):
            x = np.hstack((np.zeros(x.shape[0]*self.padding).reshape(-1,self.padding),x)) # left 
            x = np.hstack((x,np.zeros(x.shape[0]*self.padding).reshape(-1,self.padding)))# right
            x = np.vstack((x,np.zeros(x.shape[1]*self.padding).reshape(self.padding,-1)))# top
            x = np.vstack((np.zeros(x.shape[1]*self.padding).reshape(self.padding,-1),x))# bottom

            return x

        return np.array([pad_1datapoint(X[i]) for i in range(X.shape[0])])

class BasePooling2d(BasePadding2d):
    def __init__(self,pool_size=(3,3),padding=0,stride=1):
        assert len(pool_size)==2, "Pool size must be tuple of size 2"
        assert isinstance(padding,int) and padding >= 0, "Padding must an int >= 0"
        self.pool_size = pool_size
        self.padding = padding

        assert stride > 0, "Stride must be at min 1"
        if stride > 1 :
            raise NotImplementedError("Stride Greater than 1 is not implemented")
        self.stride = stride
    
           
    def plug(self,inputlayer):
        assert len(inputlayer.output_shape) == 2, "Pooling 2d take only vector as input" 
        self.input_shape = inputlayer.output_shape
        self.output_shape = [1+ self.input_shape[0] + self.padding  - self.pool_size[0],1+ self.input_shape[1] + self.padding  - self.pool_size[1]]
        assert self.output_shape[0] > 0 and self.output_shape[1] > 0, "Shapes of pooling and input don't match"
        self.input_unit = inputlayer
        inputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X,*args,**kwargs):
        self.zin = self.input_unit.forward(X) 

        self.zout = self._pool(self._add_padding(self.zin))
        return self.zout
    
    def _pool(self,X,fun=np.max):
        pooled = np.zeros((X.shape[0],*self.output_shape))
        for n in range(X.shape[0]):
            for i in range(X.shape[1]-self.pool_size[0]+1):
                for j in range(X.shape[2]-self.pool_size[1]+1):
                    pooled[n,i,j] = fun(X[n,i:i+self.pool_size[0],j:j+self.pool_size[1]]) 
        
        return pooled

class MaxPooling2d(BasePooling2d,BaseLayer):

    def _pool(self,X):
        return super()._pool(X,fun=np.max)
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class MinPooling2d(BasePooling2d,BaseLayer):
    
    def _pool(self,X):
        return super()._pool(X,fun=np.min)
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta

class AvgPooling2d(BasePooling1d,BaseLayer):
    
    def _pool(self,X):
        return super()._pool(X,fun=np.mean)
               
    def backprop(self,delta,*args,**kwargs):
        # TODO
        return delta