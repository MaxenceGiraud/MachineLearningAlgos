from ..base import BaseClassifier,BaseRegressor
import numpy as np

class BaseVoting:
    def __init__(self,models):
        self.models = models

    def fit(self,X,y):
        for model in self.models :
            model.fit(X,y)
        
    def predict_all(self,X):
        res = []
        for model in self.models :
            res.append(model.predict(X))
        return res

class VotingClassifier(BaseVoting,BaseClassifier):
    def predict(self,X):
        res = np.array(self.predict_all(X))
        y_hat = []
        for i in range(X.shape[0]):
            labels,count = np.unique(res[:,i],return_counts=True)
            y_hat.append(labels[count.argmax()])
        return np.array(y_hat)

class VotingRegressor(BaseVoting,BaseRegressor):
    def predict(self,X):
        res = self.predict_all(X)
        return np.mean(res,axis=0)