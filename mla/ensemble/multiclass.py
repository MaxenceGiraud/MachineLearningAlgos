from ..base import BaseClassifier
from ..ml.leastsquares import LinearClassification
import numpy as np

def randargmax(a,**kw):
  ''' Random tie-breaking argmax
  @param a : numpy array
  '''
  return np.argmax(np.random.random(a.shape) * (a==a.max()), **kw)


class BaseMulticlass(BaseClassifier):
    def __init__(self,base_classifier=LinearClassification,base_classifier_args={}):
        self.base_classifier = base_classifier
        self.args = base_classifier_args   
        self.binaryclassifiers = None

class OneVsRestClassifier(BaseMulticlass):
    ''' One VS Rest Classifier, train several Binary classifier 
    in a one vs rest fashion to implement multiclass classification

    Parameters
    ----------
    base_classifier : class,
            Classifier used to train the model
    base_classifier_args : dict,
            Args to give to the base classifier
    '''

    def fit(self,X,y):
        self.labels = np.unique(y)
        self.binaryclassifiers = []
        
        for label in self.labels:
            idx_rest = np.where(y!=label,True,False)
            y_rest  = np.copy(y)
            y_rest[idx_rest] = -1
            y_rest[np.invert(idx_rest)] = 1
            binaryclf = self.base_classifier(**self.args) 
            binaryclf.fit(X,y_rest)
            self.binaryclassifiers.append(binaryclf)
        
    def predict(self,X):
        preds = np.zeros((len(self.binaryclassifiers),X.shape[0]))
        for i in range(len(self.binaryclassifiers)):
            preds[i] = self.binaryclassifiers[i].predict(X)
        
        y_hat = randargmax(preds,axis=0)
        y_hat = self.labels[y_hat]

        return y_hat

class OneVsOneClassifier(BaseMulticlass):
    ''' One VS ONe Classifier, train several Binary classifier 
    in a one vs one fashion to implement multiclass classification

    Parameters
    ----------
    base_classifier : class,
            Classifier used to train the model
    base_classifier_args : dict,
            Args to give to the base classifier
    '''

    def fit(self,X,y):
        self.labels = np.unique(y)
        self.n_labels = len(self.labels)
        self.binaryclassifiers = []
        
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                if i < j  :
                    idx_i = np.where(y==self.labels[i],True,False)
                    idx_j = np.where(y==self.labels[j],True,False)
                    idx_ij = idx_i+idx_j
                    y_tmp  = np.copy(y)
                    y_tmp[idx_i] = -1
                    y_tmp[idx_j] = 1

                    binaryclf = self.base_classifier(**self.args) 
                    binaryclf.fit(X[idx_ij],y_tmp[idx_ij])
                    self.binaryclassifiers.append(binaryclf)
        
    def predict(self,X):
        preds = np.zeros((len(self.labels),X.shape[0]))
        k=0
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                if i < j  :
                    preds_binary = self.binaryclassifiers[k].predict(X)
                    preds[i] -= preds_binary
                    preds[j] += preds_binary 
                    k+=1
        
        y_hat = randargmax(preds,axis=0)
        y_hat = self.labels[y_hat]

        return y_hat
            




    