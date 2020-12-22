from abc import abstractmethod
from scipy.spatial.distance import cdist

class BaseKernel:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self,x,y):
        pass

    def __add__(self,other):
        if isinstance(other,BaseKernel):
            return KernelConcat((self,other),operation=lambda a,b : a+b)
        elif isinstance(other,int) or isinstance(other,float):
            return KernelConcatFloat(self,other,operation=lambda a,b : a+b)
        else :
            raise TypeError
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __mul__(self,other):
        if isinstance(other,BaseKernel):
            return KernelConcat((self,other),operation=lambda a,b : a*b)
        elif isinstance(other,int) or isinstance(other,float):
            return KernelConcatFloat(self,other,operation=lambda a,b : a*b)
        else :
            raise TypeError
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def __sub__(self,other):
        if isinstance(other,BaseKernel):
            return KernelConcat((self,other),operation=lambda a,b : a-b)
        elif isinstance(other,int) or isinstance(other,float):
            return KernelConcatFloat(self,other,operation=lambda a,b : a-b)
        else :
            raise TypeError
    
    def __rsub__(self,other):
        return self.__sub__(other)
    
    def __div__(self,other):
        if isinstance(other,BaseKernel):
            return KernelConcat((self,other),operation=lambda a,b : a/b)
        elif isinstance(other,int) or isinstance(other,float):
            return KernelConcatFloat(self,other,operation=lambda a,b : a/b)
        else :
            raise TypeError

    def __rdiv__(self,other):
        return self.__div__(other)

    def __truediv__(self,other):
        return self.__div__(other)
    
    def __rtruediv__(self,other):
        return self.__truediv__(other)

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

        return kwargs

    def __call__(self,x,y,**kwargs):
        # Precompute 
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