from abc import abstractmethod

class BaseKernel:
    def __init__(self):
        pass

    @abstractmethod
    def f(self,x,y):
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
    
    def f(self,x,y):
        return self.operation(self.kernels[0].f(x,y),self.kernels[1].f(x,y))

class KernelConcatFloat(BaseKernel):
    def __init__(self,kernel,scale,operation):
        assert isinstance(scale,float) or isinstance(scale,int) , "Scale must be a float or an int"
        self.kernel = kernel
        self.scale = scale
        self.operation = operation
    
    def f(self,x,y):
        return self.operation(self.kernel.f(x,y),self.scale)