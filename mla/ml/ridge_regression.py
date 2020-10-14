import numpy as np

class Ridge:
    def __init__(self,lambd=1):
        self.lambd  = lambd
        self.beta = 0

    def fit(self,X,y):
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        self.beta = np.linalg.inv(self.lambd*np.identity(X.shape[1]) + X.T @ X) @ X.T @ y

    def predict(self,X):
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        return (X @ self.beta)

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse