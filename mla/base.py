import numpy as np

class BaseAny:
    def predict(self,X):
        raise NotImplementedError("This method is not supposed to be used")

class BaseRegressor(BaseAny):

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse

class BaseClassifier(BaseAny):

    def score(self,X,y):
        ''' Compute Accuracy of classifier'''
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc