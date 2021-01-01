import numpy as np

class Relu:
    def f(self,X):
        return np.maximum(X,0)
    
    def deriv(self,X):
        return np.where(X>=0,1,0)

class Sigmoid:
    def f(self,X):
        return 1 / (1 + np.exp(-X))
    
    def deriv(self,X):
        return np.exp(-X) / (1+np.exp(-X))**2

class Tanh:
    def f(self,X):
        return np.tanh(X) # (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    
    def deriv(self,X):
        return 1-np.tanh(X)**2

class Linear:
    def f(self,X):
        return X
    
    def deriv(self,X):
        return np.ones(X.shape)

class Softplus:
    def f(self,X):
        return np.log(1+np.exp(X))
    
    def deriv(self,X):
        return 1 / (1 + np.exp(-X))

class LeakyRelu:
    def __init__(self,alpha):
        self.alpha = alpha
    def f(self,X):
        return np.where(X>=0,X,self.alpha*X)
    
    def deriv(self,X):
        return np.where(X>=0,1,self.alpha)

class Softmax:
    def f(self,X):
        exps = np.exp(X- X.max(axis=1).reshape(-1,1))
        return exps / np.sum(exps,axis=1).reshape(-1,1)
    
    def deriv(self,X):
        return self.f(X) *( 1-self.f(X))

class LogActivation:
    def __init__(self,activation):
        self.activation = activation

    def f(self,X):
        return np.log(self.activation.f(X))
    
    def deriv(self,X):
        return self.activation.deriv(X) / self.activation.f(X)
