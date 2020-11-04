import numpy as np

class PolynomialRegression:
    def __init__(self,degree=2):
        self.beta = 0
        self.degree = degree

    def fit(self,X,y):
        if self.degree >= 2 :
            for d in range(2,self.degree+1):
                Xd = X**self.degree
                X = np.concatenate((X,Xd),axis=1)
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self,X):
        if self.degree >= 2 :
            for d in range(2,self.degree+1):
                Xd = X**self.degree
                X = np.concatenate((X,Xd),axis=1)
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        return (X @ self.beta)

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse

class PolynomialClassification:
    def __init__(self,degree=2):
        self.beta = 0
        self.degree = degree

    def fit(self,X,y):
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and -1
        y_new = np.zeros(y.shape)
        y_new[y == self.labels[0]] = 1
        y_new[y == self.labels[1]] = -1

        if self.degree >= 2 :
            for d in range(2,self.degree+1):
                Xd = X**self.degree
                X = np.concatenate((X,Xd),axis=1)
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y_new

    def predict(self,X): 
        if self.degree >= 2 :
            for d in range(2,self.degree+1):
                Xd = X**self.degree
                X = np.concatenate((X,Xd),axis=1)   
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # add column of 1 for the bias
        y_hat  = np.where((X @ self.beta) >0,self.labels[0],self.labels[1])
        return y_hat

    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc