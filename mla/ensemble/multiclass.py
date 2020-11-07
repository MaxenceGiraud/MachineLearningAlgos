from ..base import BaseClassifier
from ..ml.leastsquares import LinearClassification
import numpy as np

def randargmax(a,**kw):
  ''' Random tie-breaking argmax
  @param a : numpy array
  '''
  return np.argmax(np.random.random(a.shape) * (a==a.max()), **kw)

class OneVsRestClassifier(BaseClassifier):
    ''' One VS Rest Classifier, train several Binary classifier 
    in a one vs rest fashion to implement multiclass classification

    Parameters
    ----------
    base_classifier : class,
            Classifier used to train the model
    base_classifier_args : dict,
            Args to give to the base classifier
    '''

    def __init__(self,base_classifier=LinearClassification,base_classifier_args={}):
        self.base_classifier = base_classifier
        self.args = base_classifier_args   
        self.binaryclassifiers = []

    def fit(self,X,y):
        self.labels = np.unique(y)[0]
        self.binaryclassifiers = []
        
        for label in self.labels:
            idx_rest = np.where(y!=label,True,False)
            y_rest  = y
            y_rest[idx_rest] = -1
            y_rest[not idx_rest] = 1

            binaryclf = self.base_classifier(**self.args) 
            binaryclf.fit(X,y_rest)
            self.binaryclassifiers.append(binaryclf)
        
    def predict(self,X):
        preds = np.zeros((X.shape[0],len(self.binaryclassifiers)))
        for i in range(len(self.binaryclassifiers)):
            preds[i] = self.binaryclassifiers[i].predict(X)
        
        y_hat = randargmax(preds,axis=1)
        y_hat = self.labels[y_hat]

        return y_hat
            




    