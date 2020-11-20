import numpy as np
from ..base import BaseClassifier


class BernoulliNaiveBayes(BaseClassifier):
    ''' Bernouilly naive Bayes classifier 
    Parameters
    ----------
    encoding  : string,
        Type of input data to consider: categorial (Boolean Matrix) or object (ex : list of words)
    '''

    def __init__(self,encoding="categorical"):
        assert (encoding in ["object","categorical"]), "Encoding must either be object or categorical"

        self.cond_prob = None
        self.prior = None
        self.features = None
        self.labels  = None
        self.encoding = encoding
    
    def fit(self,X,y):
        '''
        if encoding = object :
            X : list of list of object
            y  : list of corresponding labels

        elif encoding = categorical :
            X : numpy array of boolean 
            y : list of corresponding labels
        '''
        self.labels,Nk = np.unique(y,return_counts=True)
        self.prior = Nk/len(y) # Called rho_{k} in ML2

        if self.encoding == 'object' : 
            self.features = list(set([ xi for x in X for xi in x])) # List of unique features in X

            #Computing the conditionnal probabilities
            self.cond_prob = dict()  # Called Pi_{ki} in ML2
            i=0
            for l in self.labels : # for each label
                Xk = np.array(X)[np.array(y) == l] # training features corresponding to this label
                self.cond_prob[l] = dict()
                for f in self.features : self.cond_prob[l][f]=(1./ (Nk[i]+2)) # init features count

                for x in Xk :
                    for f in np.unique(x) : 
                        self.cond_prob[l][f] += (1./ (Nk[i]+2))

                i+=1

        elif self.encoding == 'categorical' :
            self.cond_prob = np.zeros((len(self.labels),X.shape[1]))

            for i in range(len(self.labels)) :
                Xl = X[np.array(y) == self.labels[i]]

                self.cond_prob[i] = (1+np.sum(Xl,axis=0))/(Nk[i]+2)
    
    def predict(self,X):
        
        #assert (self.cond_prob != None), "Model has not been trained"

        if self.encoding == 'object' :
            y_hat = []
            for x in X :
                probs = []
                for i in range(len(self.labels)):
                    l = self.labels[i]
                    prob_k = np.log(self.prior[i]) # Called y_k in ML2, add the log of prior

                    for f in self.features : 
                        if f in x :
                            prob_k += np.log(self.cond_prob[l][f])
                        else : 
                            prob_k += np.log(1. - self.cond_prob[l][f])
                    probs.append(prob_k)
                y_hat.append(self.labels[np.argmax(probs)])
        
        elif self.encoding == 'categorical':
            # see sklearn.utils.extmath.safe_sparse_dot
            probs  = np.log(self.prior) + X @ np.log(self.cond_prob.T)# + (1-X)@np.log(1-self.cond_prob.T) 
            y_hat = self.labels[np.argmax(probs,axis=1)]

        return y_hat

        
class GaussianNaiveBayes(BaseClassifier):
    ''' Gaussian naive Bayes classifier
    Parameters
    ----------
    var_smoothing : float, default=0.001
      Smoothing parameter (0 for no smoothing).
    '''
    def __init__(self,var_smoothing=1e-9):
        self.prior = None
        self.labels  = None
        self.mu = None
        self.std = None
        self.var_smoothing = var_smoothing
    
    def fit(self,X,y):
        self.labels,Nk = np.unique(y,return_counts=True)
        self.prior = Nk/len(y) 

        self.mu = np.zeros((len(self.labels),X.shape[1]))
        self.std = np.zeros((len(self.labels),X.shape[1]))
        for i in range(len(self.labels)) : 
            Xl = X[np.array(y) == self.labels[i]]

            self.mu[i] = np.sum(Xl,axis=0) / Nk[i]

            #In some case may need to add some smoothing (see sklearn)
            self.std[i] = np.std(Xl,axis=0)**2 + self.var_smoothing

    def predict(self,X):

        probs = np.zeros((X.shape[0],len(self.labels))) +np.log(self.prior) 

        for i in range(len(self.labels)):
            probs[:,i] += - 0.5 *np.sum( np.log(2*np.pi*self.std[i]) + (X-self.mu[i])**2/self.std[i],axis=1)
        
        y_hat = self.labels[np.argmax(probs,axis=1)]
        return y_hat


class MultinomialNaiveBayes(BaseClassifier):
    '''  Multinomial naive Bayes classifier 
    Parameters
    ----------
    alpha : float, default=0.001
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        '''
    def __init__(self,alpha=0.001):
        self.alpha = alpha

    def fit(self,X,y):

        self.labels,Nk = np.unique(y,return_counts=True)
        self.prior = Nk/len(y) # Called rho_{k} in ML2

        self.cond_prob = np.zeros((len(self.labels),X.shape[1]))

        for i in range(len(self.labels)) :
            Xl = X[np.array(y) == self.labels[i]]

            self.cond_prob[i] = (self.alpha+np.sum(Xl,axis=0))/(Nk[i]+X.shape[1]+1)

    def predict(self,X):
        probs  = np.log(self.prior) + X @ np.log(self.cond_prob.T)
        y_hat = self.labels[np.argmax(probs,axis=1)]
        
        return y_hat
