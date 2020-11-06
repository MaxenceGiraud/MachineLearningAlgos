import numpy as np

def relu(X):
    return np.maximum(X,0)

def relu_d(X):
    return np.where(X>=0,1,0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_d(X):
    return np.exp(-X) / (1+np.exp(-X))**2

def tanh(X):
    return np.tanh(X) # (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def tanh_d(X):
    return 1-np.tanh(X)**2

def linear(X):
    return X

def linear_d(X):
    return np.ones(X.size)

def softplus(X):
    return np.log(1+np.exp(X))

def softplus_d(X):
    return 1 / (1 + np.exp(-X))

def leakyrelu(X):
    return np.where(X>=0,X,0.001*X)

def leakyrelu_d(X):
    return np.where(X>=0,1,0.001)