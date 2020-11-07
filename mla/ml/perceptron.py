import numpy as np
from ..base import BaseClassifier

def str_to_function(arg):
    switcher = { 
        "sigmoid": (sgm,0.5), 
        'tanh' : (tanh,0),
    } 
    if arg not in switcher :
        raise Exception("Non valid activation function")
    return switcher.get(arg,None) 

def sgm(x):
    return (1 / (1 + np.exp(-x)))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


class Perceptron(BaseClassifier):
    def __init__(self,learning_rate = 0.1,theta0 = None,batch_size=32 ,activation="sigmoid",epsilon=1e-4,iter_max=200):
        if isinstance(activation,str) :
            self.activation,self.threshold = str_to_function(activation)
        else : 
            raise Exception("activation must either be a string with value in ['sigmoid','tanh']")
        
        assert batch_size >= 2, "Batch Size must be greater or equal to 2"
            
        self.lr = learning_rate
        self.theta = theta0
        self.batch_size = batch_size
        self.eps = epsilon
        self.thetas = []
        self.iter_max = iter_max

    def fit(self,X,y):
        if np.all(self.theta == None): 
            self.theta = np.ones(X.shape[1]+1)
            self.thetas.append(self.theta)
        
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and 0
        y_new = np.copy(y)
        y_new[y == self.labels[0]] = 1
        y_new[y == self.labels[1]] = 0

        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias/intercept

        grad = 1
        grad_old = 0
        nb_iter = 0

        while np.linalg.norm(grad-grad_old)/np.linalg.norm(grad) > self.eps and nb_iter < self.iter_max :

            # Random permutation
            permut = np.random.permutation(X.shape[0])
            X = X[permut]
            y = y[permut]
            
            for i in range(len(y) // self.batch_size):
                grad_old = grad
                range_start,range_end = i*self.batch_size,(i+1)*self.batch_size

                pred =  self.activation(X[range_start:range_end] @  self.theta)
                err = pred - y[range_start:range_end]
                grad = X[range_start:range_end].T @  err
                self.theta = self.theta-(self.lr*grad)

                self.thetas.append(self.theta)      

            # last mini batch      
            grad_old = grad

            pred =  self.activation(X[i*self.batch_size:] @  self.theta)
            if nb_iter == self.iter_max:
                print(pred)
            err = pred - y[i*self.batch_size:]
            grad = X[i*self.batch_size:].T @  err
            self.theta = self.theta-(self.lr*grad)

            self.thetas.append(self.theta)    

            nb_iter += 1


    def predict_probs(self,X):
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias/intercept
        return self.activation(X @ self.theta)
    
    def predict(self,X):
        probs = self.predict_probs(X)
        return np.where(probs>self.threshold,self.labels[0],self.labels[1])