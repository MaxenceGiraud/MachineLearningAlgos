import numpy as np

class BernoulliNaiveBayes:
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
            probs  = np.log(self.prior) + X @ np.log(self.cond_prob.T) + (1-X)@np.log(1-self.cond_prob.T) 
            y_hat = self.labels[np.argmax(probs,axis=1)]

        return y_hat

    
    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc

        
class GaussianNaiveBayes:
    def __init__(self):
        pass

class MultinomialNaiveBayes:
    def __init_(self):
        pass