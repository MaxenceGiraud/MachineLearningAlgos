import numpy as np

class LDA:
    'LDA Classifier'''
    def __init__(self):
        self.sigma = 0
        self.pi = 0
        self.mu = 0
        self.labels = 0

    def fit(self,X,y):
        self.labels,Nk = np.unique(y, return_counts=True) ## Computing Nk
        self.pi = Nk / np.sum(Nk)      ## pi = Nk / N

        xn = [X[lidx] for lidx in [[y==l] for l in self.labels]]

        sum_x = np.vstack([np.sum(xn[f],axis=0) for f in range(len(self.labels))])
        self.mu = 1/Nk * sum_x.T 

        for f in range(len(self.labels)):
            xmu_f = xn[f]-self.mu[:,f].T
            sigma_f = 1/Nk[f] * (xmu_f.T @ xmu_f)
            #sigma_k.append(sigma_f)
            self.sigma += sigma_f * self.pi[f]


    def predict(self,X):
        def y_lda(x,sigma,mu,pi):
            '''
            Compute yk of LDA using all parameters
            '''
            wk = np.linalg.inv(sigma) @ mu
            wk0 = -1/2 * mu.T @ np.linalg.inv(sigma) @ mu + np.log(pi)
            yk = x @ wk + wk0
            return yk

        yk = [y_lda(X,self.sigma,self.mu[:,f],self.pi[f]) for f in range(len(self.labels))]

        deciding = np.argmax(yk,axis = 0) # biggest y for each X

        y_hat = self.labels[deciding]

        return y_hat

    def score(self,X,y):
        '''Compute Accuracy'''
        y_hat = self.predict(X)
        errors  = np.count_nonzero(y_hat-y)

        acc = 1- (errors / len(y))
        return acc
