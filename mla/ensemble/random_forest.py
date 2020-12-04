import multiprocessing as mp
from functools import partial 
import numpy as np
from ..base import BaseClassifier,BaseRegressor
from ..ml.decison_tree import DecisionTreeClassifier,DecisionTreeRegressor

class Base_Randomforest:
    def __init__(self,basetree,basetree_params={},n_tree = 20,samples_ratio=0.6,parallelize=False):
        self.basetree = basetree
        self.basetree_params = basetree_params
        self.parallelize = parallelize
        self.n_tree = n_tree
        self.samples_ratio = samples_ratio
        
        self.estimators = []

    def fit(self,X,y):
        
        # Draw bootstrap samples
        n_sample = int(X.shape[0] * self.samples_ratio)
        samples_idx = np.random.choice(X.shape[0],size=self.n_tree*n_sample ,replace=True).reshape(self.n_tree,n_sample)

        # Create base tree
        self.estimators = []
        [self.estimators.append(self.basetree(**self.basetree_params)) for _ in range(self.n_tree)]

        # Fit the trees
        if self.parallelize :
            fit_func = [partial(self.estimators[i].fit,X[samples_idx[i]],y[samples_idx[i]]) for i in range(self.n_tree)]

            pool = mp.Pool(mp.cpu_count())
            [pool.apply(fit_func[i]) for i in range(self.n_tree)]

        else : 
            for i in range(self.n_tree) :
                self.estimators[i].fit(X[samples_idx[i]],y[samples_idx[i]])

    def predict_all_trees(self,X):
        if self.parallelize :
            pool = mp.Pool(mp.cpu_count())
            for i in range(self.n_tree):
                res = [pool.apply(self.estimators[i].predict,(X,)) for i in range(self.n_tree)]
        else : 
            res = []
            for i in range(self.n_tree):
                res.append(self.estimators[i].predict(X))

        return res

class RandomForestClassifier(Base_Randomforest,BaseClassifier):
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
    def __init__(self,basetree=DecisionTreeClassifier,basetree_params={},n_tree = 20,parallelize=True):
        super().__init__(basetree,basetree_params,n_tree,parallelize)

    def fit(self,X,y):
        self.labels = np.unique(y)

        return super().fit(X,y)
        
    def predict(self,X):
        res = super().predict_all_trees(X)

        # Take the most common value
        res = np.array(res)
        decision = []
        for col in range(res.shape[1]):
            values,counts = np.unique(res[:,col],return_counts=True)
            decision.append(values[counts.argmax()])
        return decision

class RandomForestRegressor(Base_Randomforest,BaseRegressor):
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
    def __init__(self,basetree=DecisionTreeRegressor,basetree_params={},n_tree = 20,parallelize=True):
        super().__init__(basetree=basetree,basetree_params=basetree_params,n_tree=n_tree,parallelize=parallelize)
    
    def predict(self,X):
        res = self.predict_all_trees(X)

        return np.mean(res,axis=0)