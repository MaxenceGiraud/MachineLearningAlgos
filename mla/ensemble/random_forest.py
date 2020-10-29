import numpy as np
import multiprocessing as mp

class Base_Randomforest:
    def __init__(self,basetree,basetree_params,n_tree = 20,parallelize=True):
        self.basetree = basetree
        self.basetree_params = basetree_params
        if parallelize :
            try :
                import multiprocessing as mp
            except :
                raise Exception("Need the multiprocessing package to parallelize")
        self.parallelize = parallelize
        self.n_tree = n_tree
        
        self.estimators = []

    def fit(self,X,y):
        
        # Draw bootstrap samples
        samples_idx = np.random.choice(X.shape[0],size=self.n_tree * X.shape[0],replace=True).reshape(self.n_tree,X.shape[0])

        # Create base tree
        self.estimators = []
        [self.estimators.append(self.basetree(**self.basetree_params)) for _ in range(self.n_tree)]

        # Fit the trees
        if self.parallelize :
            pool = mp.Pool(mp.cpu_count())
            [pool.apply(self.estimators[i].fit,(X,y))for i in range(self.n_tree)]

        else : 
            for i in range(self.n_tree) :
                self.estimators[i].fit(X[samples_idx[i]],y[samples_idx[i]])


class RandomForestClassifier(Base_Randomforest):
    ''' Random Forest Classfier
    Parameters
    ----------
    basetree : object,
            decision tree classifier object with fit and predict methods
            (may be also another base estimator)
    basetree_params : dict,
            parameters of base tree
    parallelize : bool, 
    n_tree : int,
            number of trees in the forest
    '''
    # TODO as of now class have to be in range(0,n), same with decision trees
    def fit(self,X,y):
        self.labels = np.unique(y)

        return super().fit(X,y)
        
    def predict(self,X):
        res = []
        if self.parallelize :
            pool = mp.Pool(mp.cpu_count())
            for i in range(self.n_tree):
                res.append(pool.map_async(self.estimators[i].predict,X))
        else : 
            for i in range(self.n_tree):
                res.append(self.estimators[i].predict(X))

        # Take the most common value
        res = np.array(res)
        decision = []
        for col in range(res.shape[1]):
            decision.append(np.bincount(res[:,col]).argmax())
        return decision

    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc

class RandomForestRegressor(Base_Randomforest):
    ''' Random Forest Classfier
    Parameters
    ----------
    basetree : object,
            decision tree regressor object with fit and predict methods
            (may be also another base estimator)
    basetree_params : dict,
            parameters of base tree
    parallelize : bool, 
    n_tree : int,
            number of trees in the forest
    '''
    
    def predict(self,X):
        res = []
        if self.parallelize :
            pool = mp.Pool(mp.cpu_count())
            for i in range(self.n_tree):
                res.append(pool.map_async(self.estimators[i].predict,X))
        else : 
            for i in range(self.n_tree):
                res.append(self.estimators[i].predict(X))

        return np.mean(res,axis=0)

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse