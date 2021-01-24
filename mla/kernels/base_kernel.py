from abc import abstractmethod
from scipy.spatial.distance import cdist
import numpy as np
from .metrics import chi_metric,intersection_measure

class BaseKernel:
    def __init__(self):
        pass

    def __call__(self,x,y,**kwargs):
        if y is None :
            y = x
        x,y = self._reshape(x,y)
        return self._compute_kernel(x,y,**kwargs)

    @abstractmethod
    def _compute_kernel(self,x,y,**kwargs):
        raise NotImplementedError

    def _reshape(self,x,y):
        ''' Reshape inputs x,y to 2D array if necessary'''
        if isinstance(x,int) or isinstance(x,float) or len(x.shape)<2 :
            x= np.array(x).reshape(-1,1)
        if isinstance(y,int) or isinstance(y,float) or len(y.shape)<2 :
            y= np.array(y).reshape(-1,1)
        return x,y
    
    def _create_concat(self,other,operation):
        if isinstance(other,BaseKernel):
            return KernelConcat((self,other),operation=operation)
        elif isinstance(other,int) or isinstance(other,float):
            return KernelConcatFloat(self,other,operation=operation)
        else :
            raise TypeError


    def __add__(self,other):
        print(other,type(other))
        return self._create_concat(other,operation=lambda a,b: a+b)
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __mul__(self,other):
        return self._create_concat(other,operation=lambda a,b: a*b)
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def __sub__(self,other):
        return self._create_concat(other,operation=lambda a,b: a-b)
    
    def __rsub__(self,other):
        return self._create_concat(other,operation=lambda a,b: b-a)
    
    def __div__(self,other):
        return self._create_concat(other,operation=lambda a,b: a/b)

    def __rdiv__(self,other):
        return self._create_concat(other,operation=lambda a,b: b/a)

    def __truediv__(self,other):
        return self.__div__(other)
    
    def __rtruediv__(self,other):
        return self.__rdiv__(other)

    def __pow__(self,other):
        return self._create_concat(other,operation=lambda a,b: a**b)
    
    def __rpow__(self,other):
        return self._create_concat(other,operation=lambda a,b: b**a)
    
    def __abs__(self):
        return KernelConcatFun(self,lambda a : abs(a))
    
    def exp(self):
        return KernelConcatFun(self,lambda a : np.exp(a))
    
    def log(self):
        return KernelConcatFun(self,lambda a : np.log(a))
    
    def sqrt(self):
        return KernelConcatFun(self,lambda a : np.sqrt(a))

    def tan(self):
        return KernelConcatFun(self,lambda a : np.tan(a))
    
    def tanh(self):
        return KernelConcatFun(self,lambda a : np.tanh(a))
    
    def sin(self):
        return KernelConcatFun(self,lambda a : np.sin(a))

    def cos(self):
        return KernelConcatFun(self,lambda a : np.cos(a))

    def sinh(self):
        return KernelConcatFun(self,lambda a : np.sinh(a))

    def cosh(self):
        return KernelConcatFun(self,lambda a : np.cosh(a))
    
    def arccos(self):
        return KernelConcatFun(self,lambda a : np.arccos(a))
    
    def arcsin(self):
        return KernelConcatFun(self,lambda a : np.arcsin(a))

    def arctan(self):
        return KernelConcatFun(self,lambda a : np.arctan(a))

    def arccosh(self):
        return KernelConcatFun(self,lambda a : np.arccosh(a))
    
    def arcsinh(self):
        return KernelConcatFun(self,lambda a : np.arcsinh(a))
    
    def arctanh(self):
        return KernelConcatFun(self,lambda a : np.arctanh(a))

    def apply_func(self,fn):
        return KernelConcatFun(self,lambda a : fn(a))

    def normalize(self):
        return KernelConcatFun(self,lambda a: (a-np.min(a))/np.max(a))

class KernelConcat(BaseKernel):
    def __init__(self,kernels,operation):
        self.kernels = kernels
        self.operation = operation

        self.to_precompute =kernels[0].to_precompute.union(kernels[1].to_precompute)
        self.precomputed = {}

    def get_precomputed(self,x,y,**kwargs):
        if kwargs == {} :
            if 'distance' in self.to_precompute :
                dist = cdist(x,y)
                self.precomputed['distance']= dist
            if 'inner_product' in self.to_precompute :
                prod = x @ y.T
                self.precomputed['inner_product']= prod
            if 'distance_manhattan' in self.to_precompute :
                dist = cdist(x,y,'cityblock')
                self.precomputed['distance_manhattan']= dist 
            if 'chi' in self.to_precompute :
                dist = cdist(x,y,metric=chi_metric)
                self.precomputed['chi']= dist
        return kwargs

    def __call__(self,x,y,**kwargs):
        # Precompute 
        x,y = self._reshape(x,y)
        self.precomputed = self.get_precomputed(x,y,**kwargs)

        out = self.operation(self.kernels[0](x,y,**self.precomputed),self.kernels[1](x,y,**self.precomputed))
        self.precomputed = {} # clean precompute
        return out

class KernelConcatFloat(BaseKernel):
    def __init__(self,kernel,scale,operation):
        assert isinstance(scale,float) or isinstance(scale,int) , "Scale must be a float or an int"
        self.kernel = kernel
        self.scale = scale
        self.operation = operation
        self.to_precompute = kernel.to_precompute
    
    def __call__(self,x,y,**kwargs):
        return self.operation(self.kernel(x,y,**kwargs),self.scale)

class KernelConcatFun(BaseKernel):
    def __init__(self,kernel,fn):
        self.kernel = kernel
        self.fn = fn
        self.to_precompute = kernel.to_precompute
    
    def __call__(self,x,y,**kwargs):
        return self.fn(self.kernel(x,y,**kwargs))